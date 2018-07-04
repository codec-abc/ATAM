/*!
@file		PointDetector.cpp
@brief		functions in CPointDetector
*/

#define USE_ADAPTIVE_NON_MAXIMAL_SUPPRESION 0
#include "PointDetector.h"

/*!
@brief		constructor
*/
CPointDetector::CPointDetector()
{
	// initialize detector and matcher
	mDetector = cv::ORB::create();
	mMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming(2)");
}

void adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint>& keypoints,
    const int numToKeep)
{
    if (keypoints.size() < numToKeep)
    {
        return;
    }

    //
    // Sort by response
    //
    std::sort(keypoints.begin(), keypoints.end(),
        [&](const cv::KeyPoint& lhs, const cv::KeyPoint& rhs)
    {
        return lhs.response > rhs.response;
    });

    std::vector<cv::KeyPoint> anmsPts;

    std::vector<double> radii;
    radii.resize(keypoints.size());
    std::vector<double> radiiSorted;
    radiiSorted.resize(keypoints.size());

    const float robustCoeff = 1.11f; // see paper

    for (int i = 0; i < keypoints.size(); ++i)
    {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for (int j = 0; j < i && keypoints[j].response > response; ++j)
        {
            radius = std::min(radius, cv::norm(keypoints[i].pt - keypoints[j].pt));
        }
        radii[i] = radius;
        radiiSorted[i] = radius;
    }

    std::sort(radiiSorted.begin(), radiiSorted.end(),
        [&](const double& lhs, const double& rhs)
    {
        return lhs > rhs;
    });

    const double decisionRadius = radiiSorted[numToKeep];
    for (int i = 0; i < radii.size(); ++i)
    {
        if (radii[i] >= decisionRadius)
        {
            anmsPts.push_back(keypoints[i]);
        }
    }

    anmsPts.swap(keypoints);
}

/*!
@brief		destructor
*/
CPointDetector::~CPointDetector()
{
}

/*!
@brief		initialize parameters
@param[in]	numpts		maximum number of keypoints
@param[in]	numlevel	number of pyramid levels
*/
void CPointDetector::Init(
	const int numpts,
	const int numlevel
	)
{
    _numpts = numpts;
	mDetector = 
        cv::ORB::create
        (

#if USE_ADAPTIVE_NON_MAXIMAL_SUPPRESION
            numpts * 2,
#else
            numpts,
#endif
            1.2f, 
            numlevel
        );
}

/*!
@brief		match descriptors
@param[in]	query		descriptors1
@param[in]	train		descriptors2
@param[out]	matches		matches
*/
void CPointDetector::Match(
	const cv::Mat &query,
	const cv::Mat &train,
	std::vector<cv::DMatch> &vmatch
	) const
{
	mMatcher->match(query, train, vmatch);
}

/*!
@brief		compute descriptors
@param[in]	img		gray scale image
@param[in]	vkpt	keypoints
@param[out]	desc	descriptors
*/
void CPointDetector::Describe(
	const cv::Mat &img,
	std::vector<cv::KeyPoint> &vkpt,
	cv::Mat &vdesc
	) const
{
	mDetector->compute(img, vkpt, vdesc);
}

/*!
@brief		detect keypoints
@param[in]	img		gray scale image
@param[out]	vkpt	keypoints
*/
void CPointDetector::Detect(
	const cv::Mat &img,
	std::vector<cv::KeyPoint> &vkpt
	) const
{
	mDetector->detect(img, vkpt);

#if USE_ADAPTIVE_NON_MAXIMAL_SUPPRESION
    adaptiveNonMaximalSuppresion(vkpt, _numpts);
#endif
}