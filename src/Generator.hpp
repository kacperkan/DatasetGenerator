/**
 * @file    Generator.hpp
 * @author  Filip Kocica <xkocic01@stud.fit.vutbr.cz>
 * @date    05/10/2018
 *
 * Training dataset generator
 *
 * This file provides generating of training images to backgrounds
 */

#pragma once

// STD, STL
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * @brief Class providing generating of images and annotations
 */
class DtstGenerator
{
public:
	DtstGenerator(std::ofstream& out, int imgClass);
	virtual ~DtstGenerator();

	/**
	 * @brief Generating of images & annotation
	 *
	 * @param b [in] buffer of ROIs
	 * @param m [in] background
	 * @param m2 [in] image to be copied to background
	 */
	void generate(std::vector<std::pair<cv::Point, cv::Point>>& b, cv::Mat m, cv::Mat m2);

protected:
	/**
	 * @brief Resize copied image due to its position in background
	 *
	 * @param m [in] image to be copied
	 * @param pos [in] position in background to image be copied
	 * @param mid [in] middle of background
	 */
	void resize(cv::Mat& m, int pos, int mid);
	/**
	 * @brief Create annotation file with same name as result image
	 *
	 * @param m [in] copied image
	 * @param m2 [in] background
	 * @param x [in] x position of image
	 * @param y [in] y position of image
	 */
	void createAnnotation(cv::Mat& m, cv::Mat& m2, int& x, int& y);

  /**
   * @brief
   */
  void copy2bg(cv::Mat& bg, cv::Mat& img, cv::Mat& alpha, int& x, int& y);

  /**
   * @brief
   */
  void rotateAngle(cv::Mat& img, double angle);

private:
	/** @brief Annotation output file */
	std::ofstream&  m_out;

	/** @brief Image class number */
	int             m_class;

	/** @brief RNG */
	std::mt19937    m_rng;
	std::uniform_int_distribution<std::mt19937::result_type> dist2;
	std::uniform_int_distribution<std::mt19937::result_type> dist30;
	std::uniform_int_distribution<std::mt19937::result_type> dist100;
};