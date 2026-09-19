#ifndef SKY_GLSL
#define SKY_GLSL

/**
 * Shared sky. Every consumer -- background, ambient, transparent, SSR miss -- calls this, so the
 * sky and the lighting derived from it cannot disagree.
 *
 * Daylight is the Preetham analytic model: a Perez distribution fitted to atmospheric turbidity,
 * evaluated in xyY and converted to linear sRGB. Chosen over Hosek-Wilkie because HW needs a
 * ~1500-float fitted dataset for its coefficients, where Preetham derives them from turbidity with
 * a handful of linear fits. HW renders better sunsets, so it is the upgrade path if that matters.
 *
 * Preetham is a *daylight* model and degenerates once the sun drops below the horizon, so this
 * blends to a tinted gradient through twilight. That gradient is the original night sky the lamp
 * and point-shadow work was tuned against, kept deliberately.
 */

const float SKY_PI = 3.14159265359;

// Preetham radiance is in kcd/m^2. This brings a clear midday zenith to roughly 1-3 in our linear
// HDR pipeline, where the tonemap and EV slider take over.
const float SKY_LUMINANCE_SCALE = 0.05;

/// Perez sky luminance distribution.
float skyPerez(float cosTheta, float gamma, float cosGamma, float A, float B, float C, float D, float E)
{
    return (1.0 + A * exp(B / max(cosTheta, 0.01))) * (1.0 + C * exp(D * gamma) + E * cosGamma * cosGamma);
}

/// Preetham daylight radiance for a direction. No sun disc; see skyWithSun.
vec3 skyPreetham(vec3 dir, vec3 sunDir, float T)
{
    float cosTheta = max(dir.y, 0.0);            // angle from zenith
    float cosGamma = clamp(dot(dir, sunDir), -1.0, 1.0); // angle from the sun
    float gamma = acos(cosGamma);

    float thetaS = acos(clamp(sunDir.y, 0.0, 1.0));

    // Distribution coefficients, linear in turbidity (Preetham et al. 1999, table 1).
    float AY =  0.1787 * T - 1.4630, BY = -0.3554 * T + 0.4275, CY = -0.0227 * T + 5.3251;
    float DY =  0.1206 * T - 2.5771, EY = -0.0670 * T + 0.3703;
    float Ax = -0.0193 * T - 0.2592, Bx = -0.0665 * T + 0.0008, Cx = -0.0004 * T + 0.2125;
    float Dx = -0.0641 * T - 0.8989, Ex = -0.0033 * T + 0.0452;
    float Ay = -0.0167 * T - 0.2608, By = -0.0950 * T + 0.0092, Cy = -0.0079 * T + 0.2102;
    float Dy = -0.0441 * T - 1.6537, Ey = -0.0109 * T + 0.0529;

    // Zenith luminance and chromaticity for this sun elevation.
    float chi = (4.0 / 9.0 - T / 120.0) * (SKY_PI - 2.0 * thetaS);
    float Yz = (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;

    float t2 = thetaS * thetaS, t3 = t2 * thetaS;
    float T2 = T * T;
    float xz = ( 0.00166 * t3 - 0.00375 * t2 + 0.00209 * thetaS) * T2
             + (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * thetaS + 0.00394) * T
             + ( 0.11693 * t3 - 0.21196 * t2 + 0.06052 * thetaS + 0.25886);
    float yz = ( 0.00275 * t3 - 0.00610 * t2 + 0.00317 * thetaS) * T2
             + (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * thetaS + 0.00516) * T
             + ( 0.15346 * t3 - 0.26756 * t2 + 0.06670 * thetaS + 0.26688);

    // Normalising by the zenith direction is what turns the distribution into absolute radiance.
    float denomY = skyPerez(1.0, thetaS, cos(thetaS), AY, BY, CY, DY, EY);
    float denomx = skyPerez(1.0, thetaS, cos(thetaS), Ax, Bx, Cx, Dx, Ex);
    float denomy = skyPerez(1.0, thetaS, cos(thetaS), Ay, By, Cy, Dy, Ey);

    float Y = Yz * skyPerez(cosTheta, gamma, cosGamma, AY, BY, CY, DY, EY) / max(denomY, 1e-4);
    float x = xz * skyPerez(cosTheta, gamma, cosGamma, Ax, Bx, Cx, Dx, Ex) / max(denomx, 1e-4);
    float y = yz * skyPerez(cosTheta, gamma, cosGamma, Ay, By, Cy, Dy, Ey) / max(denomy, 1e-4);

    Y = max(Y, 0.0) * SKY_LUMINANCE_SCALE;
    y = max(y, 1e-4);

    // xyY -> XYZ -> linear sRGB
    vec3 XYZ = vec3(x / y * Y, Y, (1.0 - x - y) / y * Y);
    vec3 rgb = vec3(dot(XYZ, vec3( 3.2406, -1.5372, -0.4986)),
                    dot(XYZ, vec3(-0.9689,  1.8758,  0.0415)),
                    dot(XYZ, vec3( 0.0557, -0.2040,  1.0570)));
    return max(rgb, vec3(0.0));
}

/// Three-band tinted gradient. The original night sky, kept because it is what the lamps were
/// balanced against and because Preetham has nothing to say once the sun is down.
vec3 skyNight(vec3 dir, vec3 tint)
{
    vec3 zenith = tint;
    vec3 horizon = tint * 2.2 + vec3(0.05);
    vec3 ground = tint * 0.25;

    float d = clamp(dir.y, -1.0, 1.0);
    if (d < 0.0)
    {
        return mix(horizon, ground, clamp(-d * 4.0, 0.0, 1.0));
    }
    return mix(horizon, zenith, clamp(pow(d, 0.45), 0.0, 1.0));
}

/**
 * How much daylight sky to show. Driven by sun ELEVATION and STRENGTH together -- elevation alone
 * is not enough: the sun can sit high while its intensity is dialled to 0.02 for a night scene,
 * and a Preetham midday sky over night lighting looks broken and ignores the sky tint entirely.
 *
 * The elevation term also has to finish before the horizon, because Preetham's zenith luminance
 * goes negative once the sun crosses it.
 */
float skyDayFactor(vec3 sunDir, float sunStrength)
{
    return smoothstep(-0.08, 0.15, sunDir.y) * smoothstep(0.02, 0.35, sunStrength);
}

/// Radiance without the sun disc: the one to use for ambient and reflections, since a mirror
/// ray grazing the disc would return an enormous value.
vec3 skyRadiance(vec3 dir, vec3 sunDir, float turbidity, vec3 nightTint, float sunStrength)
{
    float day = skyDayFactor(sunDir, sunStrength);
    vec3 night = skyNight(dir, nightTint);
    if (day <= 0.0)
    {
        return night;
    }
    return mix(night, skyPreetham(dir, sunDir, turbidity), day);
}

/// Sky as seen directly by the camera: radiance plus a sun disc.
vec3 skyWithSun(vec3 dir, vec3 sunDir, float turbidity, vec3 nightTint, float sunStrength)
{
    vec3 col = skyRadiance(dir, sunDir, turbidity, nightTint, sunStrength);

    // The real sun subtends about 0.53 degrees. Softened slightly at the limb so it does not alias
    // into a hard-edged polygon at this resolution.
    float cosAngle = dot(dir, sunDir);
    float disc = smoothstep(0.99985, 0.99995, cosAngle);
    return col + vec3(disc * skyDayFactor(sunDir, sunStrength) * 40.0);
}

/// Cheap two-lobe irradiance: up-facing surfaces pick up zenith, down-facing the ground tone, and
/// the reversed lobe stands in for light arriving from behind.
vec3 skyAmbient(vec3 n, vec3 sunDir, float turbidity, vec3 nightTint, float sunStrength)
{
    return mix(skyRadiance(-n, sunDir, turbidity, nightTint, sunStrength) * 0.35,
               skyRadiance(n, sunDir, turbidity, nightTint, sunStrength), 0.5);
}

#endif
