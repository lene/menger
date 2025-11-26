//==============================================================================
// Closest hit shader - computes Fresnel reflection and Snell's law refraction
//==============================================================================
extern "C" __global__ void __closesthit__ch() {
    // Get hit group data
    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // Get hit point (ray origin + t * ray direction)
    const float t = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 hit_point = ray_origin + ray_direction * t;

    // Get hit type from custom intersection program
    const unsigned int hit_kind = optixGetHitKind();
    const bool entering = (hit_kind == 0);  // 0=entry, 1=exit

    // Get surface normal from intersection attributes
    const float3 outward_normal = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2())
    );

    // Surface normal (points toward incoming ray)
    const float3 normal = entering ? outward_normal : make_float3(-outward_normal.x, -outward_normal.y, -outward_normal.z);

    // Get current depth from payload
    const unsigned int depth = optixGetPayload_3();

    // Track depth statistics
    if (params.stats) {
        atomicMax(&params.stats->max_depth_reached, depth + 1);
        atomicMin(&params.stats->min_depth_reached, depth + 1);
    }

    // Get sphere alpha from params (moved from SBT for performance)
    const float sphere_alpha = params.sphere_color[3];

    // Handle fully transparent spheres - trace continuation ray through sphere
    if (sphere_alpha < ALPHA_FULLY_TRANSPARENT_THRESHOLD) {
        // Sphere is fully transparent, let ray continue as if sphere doesn't exist
        const float3 continue_origin = hit_point + ray_direction * CONTINUATION_RAY_OFFSET;
        unsigned int continue_r = 0, continue_g = 0, continue_b = 0;
        unsigned int next_depth = depth;  // Don't increment depth for transparent pass-through

        optixTrace(
            params.handle,
            continue_origin,
            ray_direction,  // Continue in same direction
            CONTINUATION_RAY_OFFSET,
            MAX_RAY_DISTANCE,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 2, 0,  // ray_type=0 (primary), stride=2, miss_index=0
            continue_r, continue_g, continue_b, next_depth
        );

        optixSetPayload_0(continue_r);
        optixSetPayload_1(continue_g);
        optixSetPayload_2(continue_b);
        return;
    }

    // Handle fully opaque spheres (alpha >= threshold) - solid surface with diffuse shading
    if (sphere_alpha >= ALPHA_FULLY_OPAQUE_THRESHOLD) {
        // Calculate lighting (single-sided for sphere)
        const float3 lighting = calculateLighting(hit_point, normal);

        // Get sphere color from params (RGB only, ignore alpha for solid spheres)
        const float3 sphere_color = make_float3(
            params.sphere_color[0],
            params.sphere_color[1],
            params.sphere_color[2]
        );

        // Apply lighting to solid sphere surface (component-wise multiplication)
        const float3 lit_color = make_float3(
            sphere_color.x * lighting.x,
            sphere_color.y * lighting.y,
            sphere_color.z * lighting.z
        );

        const unsigned int r = static_cast<unsigned int>(fminf(lit_color.x * RayTracingConstants::COLOR_BYTE_MAX, RayTracingConstants::COLOR_BYTE_MAX));
        const unsigned int g = static_cast<unsigned int>(fminf(lit_color.y * RayTracingConstants::COLOR_BYTE_MAX, RayTracingConstants::COLOR_BYTE_MAX));
        const unsigned int b = static_cast<unsigned int>(fminf(lit_color.z * RayTracingConstants::COLOR_BYTE_MAX, RayTracingConstants::COLOR_BYTE_MAX));

        optixSetPayload_0(r);
        optixSetPayload_1(g);
        optixSetPayload_2(b);
        return;
    }

    // If max depth reached, trace one final non-recursive ray to get background/plane color
    // This avoids black artifacts from depth cutoff
    if (depth >= MAX_TRACE_DEPTH) {
        // Compute reflection direction for final ray
        const float cos_theta = fabsf(dot(ray_direction, normal));
        const float3 reflect_dir = make_float3(
            ray_direction.x - 2.0f * cos_theta * normal.x,
            ray_direction.y - 2.0f * cos_theta * normal.y,
            ray_direction.z - 2.0f * cos_theta * normal.z
        );

        // Trace one final ray at MAX_TRACE_DEPTH (will not recurse further)
        unsigned int final_r = 0, final_g = 0, final_b = 0;
        unsigned int final_depth = MAX_TRACE_DEPTH;  // Keep at max depth to prevent recursion
        const float3 final_origin = hit_point + reflect_dir * CONTINUATION_RAY_OFFSET;
        optixTrace(
            params.handle,
            final_origin,
            reflect_dir,
            CONTINUATION_RAY_OFFSET,
            MAX_RAY_DISTANCE,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 2, 0,  // ray_type=0 (primary), stride=2, miss_index=0
            final_r, final_g, final_b, final_depth
        );

        optixSetPayload_0(final_r);
        optixSetPayload_1(final_g);
        optixSetPayload_2(final_b);
        return;
    }

    // Compute Fresnel reflectance using Schlick approximation
    // Get IOR from params (moved from SBT for performance)
    const float n1 = entering ? 1.0f : params.sphere_ior;
    const float n2 = entering ? params.sphere_ior : 1.0f;
    const float r0 = (n1 - n2) / (n1 + n2);
    const float R0 = r0 * r0;
    const float cos_theta = fabsf(dot(ray_direction, normal));
    const float one_minus_cos = 1.0f - cos_theta;
    const float fresnel = R0 + (1.0f - R0) * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;

    // Compute reflection direction: r = d - 2(d·n)n
    const float3 reflect_dir = make_float3(
        ray_direction.x - 2.0f * cos_theta * normal.x,
        ray_direction.y - 2.0f * cos_theta * normal.y,
        ray_direction.z - 2.0f * cos_theta * normal.z
    );

    // Trace reflected ray
    if (params.stats) {
        atomicAdd(&params.stats->reflected_rays, 1ULL);
        atomicAdd(&params.stats->total_rays, 1ULL);
    }

    unsigned int reflect_r = 0, reflect_g = 0, reflect_b = 0;
    const float3 reflect_origin = hit_point + reflect_dir * CONTINUATION_RAY_OFFSET;
    unsigned int next_depth = depth + 1;
    optixTrace(
        params.handle,
        reflect_origin,
        reflect_dir,
        CONTINUATION_RAY_OFFSET,
        MAX_RAY_DISTANCE,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 2, 0,  // ray_type=0 (primary), stride=2, miss_index=0
        reflect_r, reflect_g, reflect_b, next_depth
    );

    // Compute refraction with Snell's law
    const float eta = n1 / n2;
    const float k = 1.0f - eta * eta * (1.0f - cos_theta * cos_theta);

    unsigned int refract_r = 0, refract_g = 0, refract_b = 0;
    if (k >= 0.0f) {
        // Refraction possible
        if (params.stats) {
            atomicAdd(&params.stats->refracted_rays, 1ULL);
            atomicAdd(&params.stats->total_rays, 1ULL);
        }

        const float coeff = eta * cos_theta - sqrtf(k);
        const float3 refract_dir = make_float3(
            eta * ray_direction.x + coeff * normal.x,
            eta * ray_direction.y + coeff * normal.y,
            eta * ray_direction.z + coeff * normal.z
        );

        const float3 refract_origin = hit_point + refract_dir * CONTINUATION_RAY_OFFSET;
        optixTrace(
            params.handle,
            refract_origin,
            refract_dir,
            CONTINUATION_RAY_OFFSET,
            MAX_RAY_DISTANCE,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 2, 0,  // ray_type=0 (primary), stride=2, miss_index=0
            refract_r, refract_g, refract_b, next_depth
        );
    } else {
        // Total internal reflection - use reflected ray result for refraction too
        refract_r = reflect_r;
        refract_g = reflect_g;
        refract_b = reflect_b;
    }

    // Apply Beer-Lambert absorption to refracted component (only at EXIT)
    float3 refract_color = make_float3(
        static_cast<float>(refract_r) / RayTracingConstants::COLOR_BYTE_MAX,
        static_cast<float>(refract_g) / RayTracingConstants::COLOR_BYTE_MAX,
        static_cast<float>(refract_b) / RayTracingConstants::COLOR_BYTE_MAX
    );

    if (!entering)  // Exiting - apply absorption to refracted ray
    {
        // Get glass color and alpha from params (moved from SBT for performance)
        const float3 glass_color = make_float3(
            params.sphere_color[0],
            params.sphere_color[1],
            params.sphere_color[2]
        );
        const float glass_alpha = params.sphere_color[3];

        // Calculate extinction constant from color and alpha
        // STANDARD GRAPHICS ALPHA CONVENTION:
        // alpha=0.0 → fully transparent (no absorption)
        // alpha=1.0 → fully opaque (maximum absorption)
        const float absorption_scale = BEER_LAMBERT_ABSORPTION_SCALE;
        const float3 extinction_constant = make_float3(
            -logf(fmaxf(glass_color.x, COLOR_CHANNEL_MIN_SAFE_VALUE)) * glass_alpha * absorption_scale,
            -logf(fmaxf(glass_color.y, COLOR_CHANNEL_MIN_SAFE_VALUE)) * glass_alpha * absorption_scale,
            -logf(fmaxf(glass_color.z, COLOR_CHANNEL_MIN_SAFE_VALUE)) * glass_alpha * absorption_scale
        );

        // Apply Beer-Lambert using t (which varies from ~0.7 at edges to ~1.2 at center)
        // Get scale from params (moved from SBT for performance)
        const float3 beer_attenuation = make_float3(
            expf(-extinction_constant.x * t * params.sphere_scale),
            expf(-extinction_constant.y * t * params.sphere_scale),
            expf(-extinction_constant.z * t * params.sphere_scale)
        );

        // Apply absorption to refracted ray
        refract_color = refract_color * beer_attenuation;
    }

    // Blend reflected and refracted rays using Fresnel coefficient
    const float3 reflect_color = make_float3(
        static_cast<float>(reflect_r) / RayTracingConstants::COLOR_BYTE_MAX,
        static_cast<float>(reflect_g) / RayTracingConstants::COLOR_BYTE_MAX,
        static_cast<float>(reflect_b) / RayTracingConstants::COLOR_BYTE_MAX
    );

    // Final color = fresnel * reflected + (1 - fresnel) * refracted
    const float3 final_color = make_float3(
        fresnel * reflect_color.x + (1.0f - fresnel) * refract_color.x,
        fresnel * reflect_color.y + (1.0f - fresnel) * refract_color.y,
        fresnel * reflect_color.z + (1.0f - fresnel) * refract_color.z
    );

    optixSetPayload_0(static_cast<unsigned int>(fminf(final_color.x * RayTracingConstants::COLOR_BYTE_MAX, RayTracingConstants::COLOR_BYTE_MAX)));
    optixSetPayload_1(static_cast<unsigned int>(fminf(final_color.y * RayTracingConstants::COLOR_BYTE_MAX, RayTracingConstants::COLOR_BYTE_MAX)));
    optixSetPayload_2(static_cast<unsigned int>(fminf(final_color.z * RayTracingConstants::COLOR_BYTE_MAX, RayTracingConstants::COLOR_BYTE_MAX)));
}
