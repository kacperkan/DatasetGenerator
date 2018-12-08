/**
 * @file    Utils.hpp
 * @author  Filip Kocica <xkocic01@stud.fit.vutbr.cz>
 * @date    12/10/2018
 *
 * Training dataset generator
 *
 * Utilities
 */

#pragma once

// STD, STL
#include <iostream>
#include <vector>
#include <experimental/filesystem>
#include <fstream>
#include <string>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * @brief Namespace which contains utils implementation
 */
namespace Utils
{
    /** @brief Buffer of images */
    using ImgBuffer = std::vector<cv::Mat>;
    /** @brief Buffer of strings */
    using StrBuffer = std::vector<std::string>;

    const std::string annotExt   = ".txt";
    const std::string imageExt   = ".jpg";
    const std::string out        = "out/";
    const std::string imgClassFn  = "imgClass";


    auto getImgClass = [](const std::string& path) -> int
    {
        std::fstream fs { path + "/" + imgClassFn, std::ios_base::in };
        int ret; fs >> ret; return(ret);
    };

    /**
     * @brief From specified path (directory) loads all images to buffer
     * 
     * @param [in] path Directory
     * @param [out] imgs Buffer of loaded images
     * @param [in] mode Mode of imread
     */
    void loadImages(const std::string& path, ImgBuffer& imgs, const int& mode);

    /**
     * @brief Arguments parser
     * 
     * @param [in] argc Number of arguments
     * @param [in] argv Argument values
     * @param [out] pathBgs Directory with backgrounds
     * @param [out] pathImg Directory with images placed into bacgrounds
     * @param [out] s Size of output images
     */
    int parseArgs(int argc, char **argv, std::string& pathBgs, std::string& pathImgs/*, cv::Size& s*/);

    /**
     * @brief Returns buffer of strings with all directories in path
     * 
     * @param [in] path Directory where we're lookin for directories
     * @param [out] strBuffer Buffer of strings with directories in path
     */
    void getDirectories(const std::string& path, StrBuffer& strBuffer);

    /**
     * @brief Prints program usage
     */
    void printUsage();
}