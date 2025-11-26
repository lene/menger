//==============================================================================
// Miss shader - returns checkered plane or background color when ray hits nothing
//==============================================================================
extern "C" __global__ void __miss__ms() {
    // Get miss data (background color)
    const MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    // Get ray data
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();

    // Extract plane parameters from params (moved from SBT for performance)
    const int plane_axis = params.plane_axis;         // 0=X, 1=Y, 2=Z
    const float plane_value = params.plane_value;     // Position along axis
    // Note: plane_positive (normal direction) is currently not used - plane is visible from both sides

    // Get ray origin/direction component for the plane axis
    float ray_orig_comp, ray_dir_comp;
    if (plane_axis == 0) {  // X axis
        ray_orig_comp = ray_origin.x;
        ray_dir_comp = ray_direction.x;
    } else if (plane_axis == 1) {  // Y axis
        ray_orig_comp = ray_origin.y;
        ray_dir_comp = ray_direction.y;
    } else {  // Z axis
        ray_orig_comp = ray_origin.z;
        ray_dir_comp = ray_direction.z;
    }

    unsigned int r, g, b;

    // Plane visible from both sides - just check for valid intersection
    // Only compute if ray direction component is non-zero
    if (fabsf(ray_dir_comp) > 1e-6f) {
        const float t = (plane_value - ray_orig_comp) / ray_dir_comp;

        if (t > 0.0f) {  // Intersection is in front of ray origin
            // Compute full intersection point
            const float3 hit_point = ray_origin + ray_direction * t;

            // For checker pattern, use the two components perpendicular to plane axis
            float checker_u, checker_v;
            if (plane_axis == 0) {  // X axis plane (YZ plane)
                checker_u = hit_point.y;
                checker_v = hit_point.z;
            } else if (plane_axis == 1) {  // Y axis plane (XZ plane)
                checker_u = hit_point.x;
                checker_v = hit_point.z;
            } else {  // Z axis plane (XY plane)
                checker_u = hit_point.x;
                checker_v = hit_point.y;
            }

            // Plane coloring: solid or checkerboard using custom colors (0.0-1.0 floats)
            float3 plane_rgb;
            if (params.plane_solid_color) {
                // Solid color from params
                plane_rgb = make_float3(
                    params.plane_color1[0],
                    params.plane_color1[1],
                    params.plane_color1[2]
                );
            } else {
                // Checkered pattern with two custom colors
                const float checker_size = PLANE_CHECKER_SIZE;
                const int check_u = static_cast<int>(floorf(checker_u / checker_size));
                const int check_v = static_cast<int>(floorf(checker_v / checker_size));

                // XOR to create checkerboard pattern
                const bool is_light = ((check_u + check_v) & 1) == 0;

                if (is_light) {
                    plane_rgb = make_float3(
                        params.plane_color1[0],
                        params.plane_color1[1],
                        params.plane_color1[2]
                    );
                } else {
                    plane_rgb = make_float3(
                        params.plane_color2[0],
                        params.plane_color2[1],
                        params.plane_color2[2]
                    );
                }
            }
            // Convert 0.0-1.0 to 0-255
            r = static_cast<unsigned int>(plane_rgb.x * COLOR_BYTE_MAX);
            g = static_cast<unsigned int>(plane_rgb.y * COLOR_BYTE_MAX);
            b = static_cast<unsigned int>(plane_rgb.z * COLOR_BYTE_MAX);

            // Apply lighting from multiple light sources to plane
            // Compute plane normal based on axis
            float3 plane_normal;
            if (plane_axis == 0) {  // X axis
                plane_normal = make_float3(1.0f, 0.0f, 0.0f);
            } else if (plane_axis == 1) {  // Y axis
                plane_normal = make_float3(0.0f, 1.0f, 0.0f);
            } else {  // Z axis
                plane_normal = make_float3(0.0f, 0.0f, 1.0f);
            }

            // Calculate lighting (double-sided for plane)
            const float3 lighting = calculateLighting(hit_point, plane_normal, true);
            r = static_cast<unsigned int>(r * lighting.x);
            g = static_cast<unsigned int>(g * lighting.y);
            b = static_cast<unsigned int>(b * lighting.z);
        } else {
            // No intersection (t < 0), use background color
            r = static_cast<unsigned int>(miss_data->r * COLOR_SCALE_FACTOR);
            g = static_cast<unsigned int>(miss_data->g * COLOR_SCALE_FACTOR);
            b = static_cast<unsigned int>(miss_data->b * COLOR_SCALE_FACTOR);
        }
    } else {
        // Ray not pointing towards plane, use background color
        r = static_cast<unsigned int>(miss_data->r * COLOR_SCALE_FACTOR);
        g = static_cast<unsigned int>(miss_data->g * COLOR_SCALE_FACTOR);
        b = static_cast<unsigned int>(miss_data->b * COLOR_SCALE_FACTOR);
    }

    // Set payload
    optixSetPayload_0(r);
    optixSetPayload_1(g);
    optixSetPayload_2(b);
}
