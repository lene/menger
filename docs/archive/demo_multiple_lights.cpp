#include "OptiXWrapper.h"
#include "OptiXData.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cmath>

void saveRGBA(const std::string& filename, int width, int height, const uint8_t* data) {
    std::string raw_file = filename + ".rgba";
    std::ofstream out(raw_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), width * height * 4);
    out.close();

    std::string cmd = "convert -size " + std::to_string(width) + "x" + std::to_string(height) +
                      " -depth 8 rgba:" + raw_file + " " + filename;
    system(cmd.c_str());
    std::remove(raw_file.c_str());
}

void normalize(float* v) {
    float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 0.0f) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

int main() {
    OptiXWrapper wrapper;
    wrapper.initialize();

    // Set up scene: opaque gray sphere and plane
    wrapper.setSphere(0.0f, 0.0f, 0.0f, 0.5f);
    wrapper.setSphereColor(0.75f, 0.75f, 0.75f, 1.0f);  // Opaque gray
    wrapper.setIOR(1.5f);
    wrapper.setScale(1.0f);
    wrapper.setPlane(1, true, -0.6f);  // Y-axis plane below sphere
    wrapper.setPlaneSolidColor(true);  // Solid light gray

    float eye[] = {0.0f, 0.0f, 5.0f};
    float lookAt[] = {0.0f, -0.3f, 0.0f};
    float up[] = {0.0f, 1.0f, 0.0f};
    wrapper.setCamera(eye, lookAt, up, 45.0f);

    int width = 800, height = 600;
    std::vector<uint8_t> output(width * height * 4);

    std::cout << "=== Multiple Lights Visual Test ===" << std::endl;
    std::cout << "Scene: Gray opaque sphere on solid plane" << std::endl << std::endl;

    // Test 1: Single directional light from top-right (default)
    {
        std::cout << "Test 1: Single directional light from top-right" << std::endl;
        float dir[] = {0.5f, 0.5f, -0.5f};
        normalize(dir);
        wrapper.setLight(dir, 1.0f);
        wrapper.setShadows(false);

        wrapper.render(width, height, output.data());
        saveRGBA("lights_1_topright_noshadow.png", width, height, output.data());
        std::cout << "  Saved: lights_1_topright_noshadow.png" << std::endl;

        wrapper.setShadows(true);
        wrapper.render(width, height, output.data());
        saveRGBA("lights_1_topright_shadow.png", width, height, output.data());
        std::cout << "  Saved: lights_1_topright_shadow.png" << std::endl;
    }

    // Test 2: Single directional light from left side
    {
        std::cout << "\nTest 2: Single directional light from left side" << std::endl;
        float dir[] = {1.0f, 0.0f, 0.0f};  // From left
        wrapper.setLight(dir, 1.0f);
        wrapper.setShadows(false);

        wrapper.render(width, height, output.data());
        saveRGBA("lights_2_left_noshadow.png", width, height, output.data());
        std::cout << "  Saved: lights_2_left_noshadow.png" << std::endl;

        wrapper.setShadows(true);
        wrapper.render(width, height, output.data());
        saveRGBA("lights_2_left_shadow.png", width, height, output.data());
        std::cout << "  Saved: lights_2_left_shadow.png" << std::endl;
    }

    // Test 3: Two directional lights (left + right)
    {
        std::cout << "\nTest 3: Two directional lights (left + right)" << std::endl;
        Light lights[2];

        // Light from left
        lights[0].type = LightType::DIRECTIONAL;
        lights[0].direction[0] = 1.0f;
        lights[0].direction[1] = 0.0f;
        lights[0].direction[2] = 0.0f;
        lights[0].color[0] = 1.0f;
        lights[0].color[1] = 1.0f;
        lights[0].color[2] = 1.0f;
        lights[0].intensity = 0.5f;

        // Light from right
        lights[1].type = LightType::DIRECTIONAL;
        lights[1].direction[0] = -1.0f;
        lights[1].direction[1] = 0.0f;
        lights[1].direction[2] = 0.0f;
        lights[1].color[0] = 1.0f;
        lights[1].color[1] = 1.0f;
        lights[1].color[2] = 1.0f;
        lights[1].intensity = 0.5f;

        wrapper.setLights(lights, 2);
        wrapper.setShadows(false);

        wrapper.render(width, height, output.data());
        saveRGBA("lights_3_leftright_noshadow.png", width, height, output.data());
        std::cout << "  Saved: lights_3_leftright_noshadow.png" << std::endl;

        wrapper.setShadows(true);
        wrapper.render(width, height, output.data());
        saveRGBA("lights_3_leftright_shadow.png", width, height, output.data());
        std::cout << "  Saved: lights_3_leftright_shadow.png" << std::endl;
    }

    // Test 4: Point light close to sphere
    {
        std::cout << "\nTest 4: Point light close to sphere" << std::endl;
        Light lights[1];

        lights[0].type = LightType::POINT;
        lights[0].position[0] = 1.5f;  // To the right
        lights[0].position[1] = 0.5f;  // Above
        lights[0].position[2] = 1.0f;  // In front
        lights[0].color[0] = 1.0f;
        lights[0].color[1] = 1.0f;
        lights[0].color[2] = 1.0f;
        lights[0].intensity = 3.0f;  // Higher intensity for point light

        wrapper.setLights(lights, 1);
        wrapper.setShadows(false);

        wrapper.render(width, height, output.data());
        saveRGBA("lights_4_point_noshadow.png", width, height, output.data());
        std::cout << "  Saved: lights_4_point_noshadow.png" << std::endl;

        wrapper.setShadows(true);
        wrapper.render(width, height, output.data());
        saveRGBA("lights_4_point_shadow.png", width, height, output.data());
        std::cout << "  Saved: lights_4_point_shadow.png" << std::endl;
    }

    // Test 5: Colored lights (red from left, blue from right)
    {
        std::cout << "\nTest 5: Colored lights (red from left, blue from right)" << std::endl;
        Light lights[2];

        // Red light from left
        lights[0].type = LightType::DIRECTIONAL;
        lights[0].direction[0] = 1.0f;
        lights[0].direction[1] = 0.0f;
        lights[0].direction[2] = 0.0f;
        lights[0].color[0] = 1.0f;  // Red
        lights[0].color[1] = 0.0f;
        lights[0].color[2] = 0.0f;
        lights[0].intensity = 0.7f;

        // Blue light from right
        lights[1].type = LightType::DIRECTIONAL;
        lights[1].direction[0] = -1.0f;
        lights[1].direction[1] = 0.0f;
        lights[1].direction[2] = 0.0f;
        lights[1].color[0] = 0.0f;
        lights[1].color[1] = 0.0f;
        lights[1].color[2] = 1.0f;  // Blue
        lights[1].intensity = 0.7f;

        wrapper.setLights(lights, 2);
        wrapper.setShadows(false);

        wrapper.render(width, height, output.data());
        saveRGBA("lights_5_colored_noshadow.png", width, height, output.data());
        std::cout << "  Saved: lights_5_colored_noshadow.png" << std::endl;

        wrapper.setShadows(true);
        wrapper.render(width, height, output.data());
        saveRGBA("lights_5_colored_shadow.png", width, height, output.data());
        std::cout << "  Saved: lights_5_colored_shadow.png" << std::endl;
    }

    // Test 6: Three lights (key + fill + rim)
    {
        std::cout << "\nTest 6: Three lights (key + fill + rim)" << std::endl;
        Light lights[3];

        // Key light (main, from top-right)
        lights[0].type = LightType::DIRECTIONAL;
        lights[0].direction[0] = 0.5f;
        lights[0].direction[1] = 0.5f;
        lights[0].direction[2] = -0.5f;
        normalize(lights[0].direction);
        lights[0].color[0] = 1.0f;
        lights[0].color[1] = 1.0f;
        lights[0].color[2] = 1.0f;
        lights[0].intensity = 0.7f;

        // Fill light (softer, from left)
        lights[1].type = LightType::DIRECTIONAL;
        lights[1].direction[0] = 1.0f;
        lights[1].direction[1] = 0.0f;
        lights[1].direction[2] = 0.0f;
        lights[1].color[0] = 1.0f;
        lights[1].color[1] = 1.0f;
        lights[1].color[2] = 1.0f;
        lights[1].intensity = 0.3f;

        // Rim light (from behind, slightly above)
        lights[2].type = LightType::DIRECTIONAL;
        lights[2].direction[0] = 0.0f;
        lights[2].direction[1] = 0.3f;
        lights[2].direction[2] = 1.0f;
        normalize(lights[2].direction);
        lights[2].color[0] = 1.0f;
        lights[2].color[1] = 1.0f;
        lights[2].color[2] = 1.0f;
        lights[2].intensity = 0.4f;

        wrapper.setLights(lights, 3);
        wrapper.setShadows(false);

        wrapper.render(width, height, output.data());
        saveRGBA("lights_6_threelights_noshadow.png", width, height, output.data());
        std::cout << "  Saved: lights_6_threelights_noshadow.png" << std::endl;

        wrapper.setShadows(true);
        wrapper.render(width, height, output.data());
        saveRGBA("lights_6_threelights_shadow.png", width, height, output.data());
        std::cout << "  Saved: lights_6_threelights_shadow.png" << std::endl;
    }

    std::cout << "\n=== All tests complete! ===" << std::endl;
    std::cout << "Check the generated PNG files to verify lighting is working." << std::endl;
    return 0;
}
