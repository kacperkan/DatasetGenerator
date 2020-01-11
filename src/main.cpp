/**
 * @file    main.cpp
 * @author  Filip Kocica <xkocic01@stud.fit.vutbr.cz>
 * @date    05/10/2018
 *
 * Training dataset generator
 *
 * Program entry point, first loads data & finds ROIs, then calls generating
 */

// Local
#include "GeneratorCropped.hpp"
#include "GeneratorTransparent.hpp"
#include "Utils.hpp"

void resizeWithRatio(cv::Mat& img, size_t height, size_t width) {
    float w = img.cols, h = img.rows;
    float ratio = w / h;
    cv::Mat newImg;
    int newWidth, newHeight;

    if (h < w) {
        newHeight = height;
        newWidth = (int)(newHeight * ratio);
    } else {
        newWidth = width;
        newHeight = newWidth / ratio;
    }
    cv::resize(img, newImg, cv::Size(newWidth, newHeight));
    img = newImg;
}

/**
 * @brief Entry point
 */
int main(int argc, char** argv) {
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Locals

    int imgClass = 0;
    int ret = 0;

    size_t randNum = 0;
    size_t randBgIndex = 0;

    std::string pathToBackgrounds;
    std::string pathToImages;
    int nImagesToGenerate;
    int maxPerBackgroundImages;

    Utils::ImgBuffer bgs;
    Utils::ImgBuffer imgs;

    Utils::StrBuffer dirs;

    cv::Mat bg;
    cv::Mat img;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Parse arguments

    if ((ret = Utils::parseArgs(argc, argv, pathToBackgrounds, pathToImages,
                                nImagesToGenerate, maxPerBackgroundImages)) !=
        0) {
        return ret;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Load images & directories of classes

    Utils::loadImages("png", pathToBackgrounds, bgs, cv::IMREAD_COLOR);
    Utils::getDirectories(pathToImages, dirs);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // RNG

    std::mt19937 rng;
    rng.seed(std::random_device()());

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Get image width & height

    std::experimental::filesystem::path mainOutputDir(Utils::outDir);
    std::experimental::filesystem::path imagesDir(pathToImages);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Iterate over all TS classes in given directory, load TS from each of them

    for (auto& path : dirs) {
        Utils::loadImages("png", path, imgs,
                          cv::IMREAD_UNCHANGED);  // With alpha-channel
    }

    size_t nBackgrounds = bgs.size();
    size_t nImages = imgs.size();

    if (nBackgrounds == 0 || nImages == 0) {
        std::cerr << "E: Failed to load images." << std::endl;
        return 1;
    }

    PRNG::Uniform_t distI = PRNG::Uniform_t(0, nImages - 1);
    PRNG::Uniform_t distB = PRNG::Uniform_t(0, nBackgrounds - 1);

    PRNG::Uniform_t distX;
    PRNG::Uniform_t distY;
    PRNG::Uniform_t distW;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Generate `nGenImgs` images for each of the TS classes

    float minPercentageOfBgWidth = 0.4;
    float maxPercentageOfBgWidth = 0.6;

    PRNG::Uniform_t perImageDist =
        PRNG::Uniform_t{1, (size_t)maxPerBackgroundImages};

    size_t minHeight = 1080;
    size_t minWidth = 1920;

    for (int imgCounter = 0; imgCounter < nImagesToGenerate; imgCounter++) {
        randBgIndex = distB(rng);
        bgs.at(randBgIndex).image.copyTo(bg);
        resizeWithRatio(bg, minHeight, minWidth);
        float viewRatio = (float)bg.cols / bg.rows;

        distW = PRNG::Uniform_t(
            (size_t)std::floor(
                std::max(minPercentageOfBgWidth * bg.cols, 1.0f)),
            (size_t)std::floor((maxPercentageOfBgWidth * bg.cols)));

        int w = (int)distW(rng);
        int h = (int)((float)w / viewRatio);

        distX = PRNG::Uniform_t{0, (size_t)(bg.cols - w - 1)};
        distY = PRNG::Uniform_t{0, (size_t)(bg.rows - h - 1)};

        int x = (int)distX(rng);
        int y = (int)distY(rng);
        std::cout << "Generating dimensions: (x, y, w, h) ";
        std::cout << x << ", " << y << ", " << w << ", " << h << std::endl;

        // Select random part of the background
        cv::Rect roi{x, y, w, h};
        bg = bg(roi);

        int imagesPerBackground = perImageDist(rng);
        // Create output annotation file passed to generator
        std::ofstream annotFile(
            mainOutputDir /
            (std::to_string(imgCounter) + Utils::antExt).c_str());

        cv::cvtColor(bg, bg, cv::COLOR_BGR2GRAY);
        cv::cvtColor(bg, bg, cv::COLOR_GRAY2BGR);

        for (int logoCount = 0; logoCount < imagesPerBackground; logoCount++) {
            randNum = distI(rng);
            imgs.at(randNum).image.copyTo(img);

            const std::string className = imgs.at(randNum).className;

            // Image & annotation generator
            imgClass = Utils::getImgClass((imagesDir / className).c_str());

            // Generate image with annotations
            DatasetGenerator_t* generator = new DatasetGeneratorTransparent_t{
                annotFile, imgClass, className};

            generator->generateDataset(bg, img);
        }

        // Save image
        cv::imwrite(
            (mainOutputDir / (std::to_string(imgCounter) + Utils::imgExt))
                .c_str(),
            bg);
    }

    // Destroy OpenCV windows if exists
#ifdef GENERATOR_DEBUG
    cv::destroyAllWindows();
#endif

    return 0;
}