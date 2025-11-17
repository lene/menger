#include "OptiXWrapper.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>

void saveRGBA(const std::string& filename, int width, int height, const uint8_t* data) {
    // Save as raw RGBA first
    std::string raw_file = filename + ".rgba";
    std::ofstream out(raw_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), width * height * 4);
    out.close();

    // Convert to PNG using ImageMagick
    std::string cmd = "convert -size " + std::to_string(width) + "x" + std::to_string(height) +
                      " -depth 8 rgba:" + raw_file + " " + filename;
    system(cmd.c_str());

    // Clean up raw file
    std::remove(raw_file.c_str());
}

int main() {
    OptiXWrapper wrapper;
    wrapper.initialize();

    // Set up scene
    wrapper.setSphere(0.0f, 0.0f, 0.0f, 0.5f);
    wrapper.setIOR(1.5f);
    wrapper.setScale(1.0f);
    wrapper.setPlane(1, true, -0.6f);

    // Camera setup
    float eye[] = {0.0f, 0.0f, 5.0f};
    float lookAt[] = {0.0f, -0.3f, 0.0f};
    float up[] = {0.0f, 1.0f, 0.0f};
    wrapper.setCamera(eye, lookAt, up, 45.0f);

    // Light setup
    float lightDir[] = {0.5f, 0.5f, -0.5f};
    wrapper.setLight(lightDir, 1.0f);
    wrapper.setShadows(true);

    int width = 800, height = 600;
    std::vector<uint8_t> output(width * height * 4);

    // Generate images for different alpha values
    float alpha_values[] = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f, 0.0f};

    std::cout << "=== Transparent Shadow Demo ===" << std::endl;

    for (float alpha : alpha_values) {
        wrapper.setSphereColor(0.75f, 0.75f, 0.75f, alpha);

        std::string filename = "demo_alpha_" + std::to_string(alpha).substr(0, 3) + ".png";
        std::cout << "Rendering alpha=" << alpha << " -> " << filename << std::endl;

        wrapper.render(width, height, output.data());
        saveRGBA(filename, width, height, output.data());
    }

    // Also render without shadows for reference
    wrapper.setShadows(false);
    wrapper.setSphereColor(0.75f, 0.75f, 0.75f, 0.5f);
    std::cout << "Rendering reference (no shadows) -> demo_no_shadows.png" << std::endl;
    wrapper.render(width, height, output.data());
    saveRGBA("demo_no_shadows.png", width, height, output.data());

    std::cout << "\nDone! Generated 7 images demonstrating transparent shadows." << std::endl;

    return 0;
}
