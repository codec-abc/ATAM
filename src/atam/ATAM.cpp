/*!
@file		ATAM.cpp
@brief		functions in CATAM
*/

#include "ATAM.h"
#include "Timer.h"

#include <numeric> 
#include <fstream>

extern sATAMParams PARAMS;		//!< parameters in ATAM

/*!
@brief		constructor
*/
CATAM::CATAM()
{
#if DO_BA
    mDoingBA = false;
#endif
    reset();
}

/*!
@brief		start ATAM
*/
void CATAM::Start(void)
{
    // initialization
    if (!init())
    {
        return;
    }

    // start
#ifdef MULTITHREAD
    #if DO_BA && !ONLY_DO_INITIAL_BA
            std::thread BA(&CATAM::BA, this);
            mainLoop();
            BA.join();
    #else
            mainLoop();
    #endif
#else
    mainLoop();
#endif
}


/*!
@brief		initialization
@retval		succeeded or not
*/
bool CATAM::init(void)
{
    // load parameters
    PARAMS.loadParams(strData + "params.xml");

    // open camera
    int width, height, channel;
    if (PARAMS.USEVIDEO)
    {
        if (!mCam.OpenVideo(PARAMS.VIDEONAME, width, height, channel))
        {
            printf("Cannot open %s\n", PARAMS.VIDEONAME.c_str());
            return false;
        }
    }
    else
    {
        if (!mCam.Open(width, height, channel))
        {
            printf("Cannot open camera\n");
            return false;
        }
    }

    // load camera parameters
    if (!mCam.LoadParameters(PARAMS.CAMERANAME))
    {
        printf("Cannot open %s\n", PARAMS.CAMERANAME.c_str());
        return false;
    }

    // set camera parameters to ATAM
    mCam.A.copyTo(mData.A);					// camera parameters	
    mCam.D.copyTo(mData.D);					// distortion parameters
    mData.focal = mCam.A.at<double>(0, 0);	// focal length

    // prepare image memory
    mImg = cv::Mat(cv::Size(width, height), CV_8UC3);
    mGImg = cv::Mat(cv::Size(width, height), CV_8UC1);

    // load challenge points if exist
    loadChallenge(strData + "challenge.txt");

    // initialize keypoint detector
    mDetector.Init(PARAMS.MAXPTS, PARAMS.LEVEL);

    // generate button
    generateButton();

    return true;
}

/*!
@brief		main loop
*/
void CATAM::mainLoop(void)
{
    // create window and set mouse callback for OpenCV
    const char windowName[128] = "ATAM";
    cv::namedWindow(windowName);
    cv::setMouseCallback(windowName, mousedummy, this);

    // timer
    CTimer timer;

    while (1)
    {

        timer.Push(__FUNCTION__);

        // get color image and convert it to gray scale
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
        mCam.Get(mImg);
        cv::cvtColor(mImg, mGImg, cv::COLOR_BGR2GRAY);
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto frame_duration = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000.0f;

        printf("image retrieval and convert took : %f ms \n ", frame_duration);

        // process

        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
#endif
        process();
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
        frame_duration = (std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000.0f;
        //printf( "process took : %f\n", frame_duration);
        puts("\n\n");
#endif
        // get FPS
        double duration = double(timer.Pop());
        mFPS = 1.0 / duration * 1000.0;
        //LOGOUT("--frame %d--\n", mFrameNumber);
        ++mFrameNumber;

        // get keyboard input
        int mouse = mMouse;

        // user's operation
        if (operation(cv::waitKey(1)))
        {
            break;
        }

        // show image
        draw(mImg);

        // show image
        cv::imshow(windowName, mImg);
    }

    // close
    cv::destroyWindow(windowName);
    mState = STATE::CLOSE;
}


/*!
@brief			main process
*/
void CATAM::process(void)
{
    // detect keypoint
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    if (_needReset)
    {
        printf( "BA error too high. Resetting\n");
        reset();
        return;
    }
    mData.vKpt.clear();
    mDetector.Detect(mGImg, mData.vKpt);
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000.0f;
    printf("[process] keypoints clear & detector detect : %f ms \n", duration_ms);
#endif

    // each process
    switch (mState)
    {
    case STATE::INIT:
        whileInitialize();
        break;
    case STATE::TAM:
    {
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
#endif
        int nbFeatureTrackedPoint;
        int trackedPoints = trackAndMap(nbFeatureTrackedPoint);
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
        auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000.0f;
        printf( "[process] trackAndMap took : %f ms\n", duration_ms);
#endif
        if (!mData.haveScale)
        {
            cv::Mat rvec, tvec;
            double reproError = 0;
            bool found = mCalibrator.EstimatePose(mGImg, mData.A, mData.D, rvec, tvec, reproError);

            int totalPoints = 0;
            int mappedPoints = 0;
            int newPoints = 0;

            getPointsCount(totalPoints, mappedPoints, newPoints);

            printf
            (
                "[3] Features points : %i tracked points : %i\n",
                newPoints,
                mappedPoints
            );

            if (mappedPoints < PARAMS.RESET_BELOW_NB_POINTS)
            {
                reset();
                return;
            }

            if (found && mData.vPosePair.size() == 0 && mappedPoints > PARAMS.MIN_FEATURES_POINT && reproError < PARAMS.MAX_REPRO_ERROR)
            {
                printf
                (
                    "3- Change state at %i, reprojection error %f \n",
                    mFrameNumber,
                    reproError
                );

                registerWorld();

                _x = tvec.at<double>(0);
                _y = tvec.at<double>(1);
                _z = tvec.at<double>(2);

            }
            else if (found && nbFeatureTrackedPoint > PARAMS.MIN_FEATURES_POINT && reproError < PARAMS.MAX_REPRO_ERROR)
            {
                double x = tvec.at<double>(0);
                double y = tvec.at<double>(1);
                double z = tvec.at<double>(2);

                double dist = sqrt((x - _x) * (x - _x) + (y - _y) * (y - _y));//  +(z - _z) * (z - _z)

                if (dist > PARAMS.MIN_STEREO_DIST)
                {
                    printf
                    (
                        "4- Change state at %i, dist is %f\n",
                        mFrameNumber,
                        dist
                    );
                    registerWorld();

                }
            }
        }
        else
        {
            int totalPoints = 0;
            int mappedPoints = 0;
            int newPoints = 0;

            getPointsCount(totalPoints, mappedPoints, newPoints);

            if (mappedPoints < PARAMS.RESET_BELOW_NB_POINTS)
            {
                printf
                (
                    "RESETTING because not enough tracked points %i expected %i\n",
                    mappedPoints,
                    PARAMS.RESET_BELOW_NB_POINTS
                );

                reset();
                return;
            }
        }
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
        duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count()) / 1000.0f;
        printf( "[process] inner case 2nd part took : %f ms \n", duration_ms);
#endif
    }
        break;
    case STATE::RELOCAL:
        relocalize();
        break;
    case STATE::STOP:
        if (mImg.size != 0 && _lastFrameAction != mFrameNumber)
        {
            startInit();
        }
        break;
    default:
        break;
    }

    // keep image for tracking in next image
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
#endif
    mGImg.copyTo(mData.prevGImg);

#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();
    /*
    duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count()) / 1000.0f;
    printf( "[process] mGimgCopy : %f ms \n", duration_ms);
    */
    auto duration_ms_ = (std::chrono::duration_cast<std::chrono::microseconds>(t6 - t1).count()) / 1000.0f;;
    printf( "[process] total : %f ms \n", duration_ms_);
#endif
}

/*!
@brief		start initialization
*/
void CATAM::startInit(void)
{
    // set initial pose
    mPose.rvec.setTo(0);
    mPose.tvec.setTo(0);

    cv::Mat rvec, tvec;
    double reproError = 0;
    bool found = mCalibrator.EstimatePose(mGImg, mData.A, mData.D, rvec, tvec, reproError);
    if (found && reproError < PARAMS.MAX_REPRO_ERROR)
    {
        _x = tvec.at<double>(0);
        _y = tvec.at<double>(1);
        _z = tvec.at<double>(2);

        // start keyframe
        if (!setKeyframe())
        {
            reset();
            return;
        }
        else
        {
            int totalPoints = 0;
            int mappedPoints = 0;
            int newPoints = 0;

            getPointsCount(totalPoints, mappedPoints, newPoints);

            printf
            (
                "[1] Features points : %i tracked points : %i\n",
                newPoints,
                mappedPoints
            );

            if (newPoints < PARAMS.MIN_FEATURES_POINT || reproError >= PARAMS.MAX_REPRO_ERROR)
            {
                reset();
                return;
            }
            else
            {
                // change state   
                mState = STATE::INIT;
                mText = "Translate camera and press space";

                _lastFrameAction = mFrameNumber;

                printf
                (
                    "1- Change state at %i, reprojection error %f\n",
                    mFrameNumber,
                    reproError
                );

                _tvecInit1 = tvec;
                _rvecInit1 = rvec;
            }
        }
    }
    else
    {
        reset();
    }
}

/*!
@brief		start tracking and mapping
*/
void CATAM::startTAM(void)
{
    // if initial map is successfully generated
    //if (makeMap())
    cv::Mat rvec, tvec;
    double reproError = 0;
    bool found = mCalibrator.EstimatePose(mGImg, mData.A, mData.D, rvec, tvec, reproError);
    if (makeMap() && found && _lastFrameAction != mFrameNumber && reproError < PARAMS.MAX_REPRO_ERROR)
    {
        printf("startTAM with reprojection error %f\n", reproError);

        // set keyframe
        setKeyframe();

        // change state
        mState = STATE::TAM;
        mText = "Capture calibration board and press space";

        _lastFrameAction = mFrameNumber;
    }
    else
    {
        reset();
    }
}

/*!
@brief		change state
*/
void CATAM::changeState(void)
{
    if (mState == STATE::STOP)
    {
        startInit();		// initialize
    }
    else if (mState == STATE::INIT)
    {
        startTAM();			// start tracking and mapping
    }
    else if (mState == STATE::TAM)
    {
        registerWorld();	// registration with world coordinate system
    }
}

/*!
@brief		reset
*/
void CATAM::reset(void)
{
    mState = STATE::STOP;

    mFrameNumber = 0;
    mFPS = 0.0;
    mChallengeNumber = 0;
    _lastFrameAction = 0;

    _x = 0;
    _y = 0;
    _z = 0;

    _needReset = false;

#if DO_BA && !ONLY_DO_INITIAL_BA
        // stop BA
        bool tmp = true;
        while (tmp)
        {
            mBAMutex.lock();
            tmp = mDoingBA;
            mBAMutex.unlock();
        }
#endif

    mData.clear();

    // change state
    mText = "Press space to start";
    //LOGOUT("-------------RESET-------------\n");
}


/*!
@brief		set keyframe for tracking
@retval		set or not
*/
bool CATAM::setKeyframe(void)
{
    // remove points not mapped 
    mData.clearTrack(NOID);

    // prepare keyframe
    sKeyframe tmpKf;
    tmpKf.pose = mPose;

    // set points in keyframe
    for (std::list<sTrack>::iterator it = mData.vTrack.begin(),
        itend = mData.vTrack.end(); it != itend; ++it)
    {
        tmpKf.vPt.push_back(it->vPt.back());
        tmpKf.vPtID.push_back(it->ptID);
    }

    // check keypoint already exists in tracks or not
    std::vector<int> vNewptID;
    std::vector<cv::KeyPoint> &vKpt = mData.vKpt;

    for (int i = 0, iend = int(vKpt.size()); i < iend; ++i)
    {

        bool foundSame = false;
        double minDist = PARAMS.PROJERR;
        int ID = NOID;

        for (std::list<sTrack>::iterator it = mData.vTrack.begin(),
            itend = mData.vTrack.end(); it != itend; ++it)
        {

            if (it->ptID != NOID)
            {		// for mapped point

                cv::Point2f &pt = it->vPt.back();

                double dist = cv::norm(vKpt[i].pt - pt);

                if (dist < minDist)
                {
                    ID = it->ptID;
                    minDist = dist;
                    foundSame = true;
                }
            }
        }

        if (!foundSame)
        {		// if keypoints not in mapped tracks
            vNewptID.push_back(i);
        }
        else
        {					// if keypoints in mapped tracks
            tmpKf.vKpt.push_back(vKpt[i]);
            tmpKf.vKptID.push_back(ID);
        }
    }

    if (int(vNewptID.size()) < PARAMS.MINPTS)
    {	// not suitable if new keypoints are less
        return false;
    }

    // compute descriptor and add it to keyframe
    mGImg.copyTo(tmpKf.img);
    cv::Mat vDesc;
    if (tmpKf.vKpt.size() != 0)
    {
        mDetector.Describe(tmpKf.img, tmpKf.vKpt, tmpKf.vDesc);
    }
    mData.map.AddKeyframe(tmpKf);

    // add new keypoints in new tracks
    for (int i = 0, iend = int(vNewptID.size()); i < iend; ++i)
    {
        sTrack tmpTrack;
        tmpTrack.kpt = vKpt[vNewptID[i]];
        tmpTrack.ptID = NOID;
        mData.addTrack(tmpTrack);
    }

    return true;
}

/*!
@brief		check inside image or not
@param[in]	pt	point
@retval		inside or not
*/
bool CATAM::checkInsideImage(const cv::Point2f &pt) const
{
    int space = PARAMS.PATCHSIZE * 2;

    if (pt.x < space
        || mGImg.cols - space < pt.x
        || pt.y < space
        || mGImg.rows - space < pt.y)
    {
        return false;
    }
    else
    {
        return true;
    }
}


/*!
@brief		track points between frames using KLT
@retval		number of tracked mapped points
*/
int CATAM::trackFrame(int &nbFeaturePoints)
{
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif

    std::vector<cv::Point2f> vTracked;
    std::vector<unsigned char> vStatus;
    std::vector<float> vError;
    const cv::Size patch(PARAMS.PATCHSIZE, PARAMS.PATCHSIZE);

    cv::TermCriteria criteria
    (
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, // type
        10, // maxcount
        0.1 // epsilon
    );

    // KLT
    cv::calcOpticalFlowPyrLK
    (
        mData.prevGImg, 
        mGImg, 
        mData.vPrevPt, 
        vTracked, 
        vStatus, 
        vError, 
        patch,
        3 /*maxLevel*/,
        criteria /*criteria*/,
        0 /*flag*/,
        0.01 /*minEigThreshold*/
    );
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000.0f;
    printf( "    [trackFrame] calcOpticalFlowPyrLK : %f ms \n", duration_ms);

    // remove point from tracks if point not tracked
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
#endif
    int count = 0;
    mData.vPrevPt.clear();

    std::list<sTrack>::iterator it = mData.vTrack.begin();
    nbFeaturePoints = 0;
    for (size_t i = 0; i < vStatus.size(); ++i)
    {
        if (!vStatus[i] || !checkInsideImage(vTracked[i]))
        {
            it = mData.vTrack.erase(it);
        }
        else
        {
            nbFeaturePoints++;
            mData.vPrevPt.push_back(vTracked[i]);
            it->vPt.push_back(vTracked[i]);

            if (it->ptID != NOID)
            {
                ++count;
            }
            ++it;
        }
    }

#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000.0f;
    printf("    [trackFrame] remove point from tracks if point not tracked took %f ms\n", duration_ms);
#endif
    return count;
}

/*!
@brief		match with keyframe near mPose and recover points
@retval		matched or not
*/
bool CATAM::matchKeyframe(void)
{
    // compute descriptors of current frame
    std::vector<cv::KeyPoint> &vKpt = mData.vKpt;
    if (vKpt.size() < PARAMS.MINPTS)
    {
        return false;
    }
    cv::Mat vDesc;
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    mDetector.Describe(mGImg, vKpt, vDesc);
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000.0f;
    printf( "    [matchKeyFrame] detector describe : %f ms \n", duration_ms);
#endif

    // get nearest keyframe
    const sKeyframe& kf = mData.map.GetNearestKeyframe(mPose);
    if (kf.vKpt.size() < PARAMS.MINPTS)
    {		// not enough keypoint in keyframe
        return false;
    }

    // matching
    std::vector<cv::DMatch> vMatch;
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
#endif
    mDetector.Match(vDesc, kf.vDesc, vMatch);

#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000.0f;
    printf( "    [matchKeyFrame] detector match : %f ms \n", duration_ms);
#endif

    std::vector<cv::Point3f> vPt3d;
    std::vector<cv::Point2f> vPt2d;
    std::vector<int> vID;

    // select good matches
    for (int i = 0, iend = int(vMatch.size()); i < iend; ++i)
    {
        if (vMatch[i].distance < PARAMS.DESCDIST)
        {
            vPt2d.push_back(vKpt[vMatch[i].queryIdx].pt);

            int ID = vMatch[i].trainIdx;
            vPt3d.push_back(mData.map.GetPoint(kf.vKptID[ID]));
            vID.push_back(kf.vKptID[ID]);
        }
    }

#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
    duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count()) / 1000.0f;
    printf( "    [matchKeyFrame] selecting good matches took : %f ms \n", duration_ms);
#endif

    if (vPt2d.size() > PARAMS.MINPTS)
    {	// if enough good correspondences

        // compute camera pose
        sPose tmpPose = mPose;
        const int iteration = 20;
        const double confidence = 0.95;
        std::vector<int> vInliers;

        std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();

        bool result = cv::solvePnPRansac
        (
            vPt3d, 
            vPt2d, 
            mData.A, 
            mData.D, 
            tmpPose.rvec, 
            tmpPose.tvec, 
            true, /* useExtrinsicGuess	Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses the provided rvec and tvec values as initial approximations of the rotation and translation vectors, respectively, and further optimizes them. */ 
            iteration, 
            PARAMS.PROJERR, 
            confidence, 
            vInliers
        );

#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();
        auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count()) / 1000.0f;
        printf( "    [matchKeyFrame] solvePnPRansac  %f ms \n", duration_ms);
#endif

        if (result != true)
        {
            return false;
        }

        // check number of inliers and inlier ratio (inlier / all)
        if 
        (
            int(vInliers.size()) > PARAMS.MINPTS && 
            float(vInliers.size()) / float(vPt2d.size()) > PARAMS.MATCHKEYFRAME
        )
        {
            std::chrono::high_resolution_clock::time_point t8 = std::chrono::high_resolution_clock::now();

            // add as a mapped track
            int numRecovered = 0;
            for (int i = 0, iend = int(vInliers.size()); i < iend; ++i)
            {

                // check already in mapped tracks
                bool found = false;
                int pos = vInliers[i];
                for (std::list<sTrack>::iterator it = mData.vTrack.begin(),
                    itend = mData.vTrack.end();	it != itend; ++it)
                {

                    if (it->ptID == vID[pos])
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    sTrack track;
                    track.kpt.pt = vPt2d[pos];
                    track.ptID = vID[pos];
                    mData.addTrack(track);
                    ++numRecovered;
                }
            }

            if (mState == STATE::RELOCAL)
            {
                mPose = tmpPose;
            }

            //LOGOUT("Recovered %d points with keyframe %d at %s\n", numRecovered, kf.ID, __FUNCTION__);

#if SLAM_PROFILING
            std::chrono::high_resolution_clock::time_point t9 = std::chrono::high_resolution_clock::now();
            duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t9 - t8).count()) / 1000.0f;
            printf( "    [matchKeyFrame] check number of inliers and inlier ratio took : %f ms \n", duration_ms);
#endif
            return true;
        }
    }

    return false;
}

/*!
@brief		compute camera pose
@retval		computed or not
*/
bool CATAM::computePose(void)
{
    // pose estimation
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    std::vector<cv::Point2f> vPt2d;
    std::vector<cv::Point3f> vPt3d;

    for (std::list<sTrack>::iterator it = mData.vTrack.begin(),
        itend = mData.vTrack.end();	it != itend; ++it)
    {
        if (it->ptID != NOID)
        {
            vPt2d.push_back(it->vPt.back());
            vPt3d.push_back(mData.map.GetPoint(it->ptID));
        }
    }

#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000.0f;
    printf("   [computePose] preparing solvePnP took %f ms\n", duration_ms);
#endif

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

    cv::solvePnP
    (
        vPt3d, 
        vPt2d, 
        mData.A, 
        mData.D, 
        mPose.rvec, 
        mPose.tvec, 
        true /* useExtrinsicsGuess */,
        cv::SOLVEPNP_ITERATIVE /* flags */
    );

#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000.0f;
    printf("   [computePose] solvePnP took %f ms\n", duration_ms);
#endif

    std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
    // check reprojection error and discard points if error is large
    std::vector< cv::Point2f > vReproPt;
    cv::projectPoints(vPt3d, mPose.rvec, mPose.tvec, mData.A, mData.D, vReproPt);

    int numAll = 0;
    int numDiscard = 0;

    for (std::list<sTrack>::iterator it = mData.vTrack.begin(),
        itend = mData.vTrack.end();	it != itend; ++it)
    {
        if (it->ptID != NOID)
        {

            double dist = cv::norm(vReproPt[numAll] - vPt2d[numAll]);

            if (dist > PARAMS.PROJERR)
            {
                it->ptID = DISCARD;
                ++numDiscard;
            }
            ++numAll;
        }
    }

    mData.clearTrack(DISCARD);

    //LOGOUT("Discarded:%d used:%d at %s\n", numDiscard, numAll - numDiscard, __FUNCTION__);

    if (mData.haveScale)
    {
        transformToWorld(mPose, mWPose);
    }
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();
    duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count()) / 1000.0f;
    printf("   [computePose] checking reprojection error and discarding points took %f ms\n", duration_ms);

    duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t6 - t1).count()) / 1000.0f;
    printf("   [computePose] whole function except last if %f ms\n", duration_ms);
#endif
    int numKept = numAll - numDiscard;

    if (numKept > PARAMS.MINPTS)
    {	// if enough points not exist
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();
        duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t7 - t1).count()) / 1000.0f;
        printf("   [computePose] whole function (case 1) took %f ms\n", duration_ms);
#endif
        return true;
    }
    else
    {
        auto nbPoints = numAll - numDiscard;
#if SLAM_PROFILING
        printf("Not enough points: %i expected %i \n", nbPoints, PARAMS.MINPTS);

        std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();
        duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t7 - t1).count()) / 1000.0f;
        printf("   [computePose] whole function (case 2) took %f ms\n", duration_ms);
#endif
        return false;
    }
}


/*!
@brief		triangulation
@param[in]	vUnP1	undistorted points1
@param[in]	vUnP2	undistorted points2
@param[in]	pose1	camera pose of vpt1
@param[in]	pose2	camera pose of vpt2
@param[out]	vpt3d		triangulated points
@retval		true or false
*/
void CATAM::triangulate
(
    const std::vector<cv::Point2f> &vUnPt1,
    const std::vector<cv::Point2f> &vUnPt2,
    const sPose &pose1,
    const sPose &pose2,
    std::vector<cv::Point3f> &vpt3d
) const
{
    cv::Mat R1, R2;

    cv::Rodrigues(pose1.rvec, R1);
    cv::Rodrigues(pose2.rvec, R2);

    cv::Mat P1(3, 4, R1.type()), P2(3, 4, R2.type());
    R1.copyTo(P1(cv::Rect(0, 0, 3, 3)));
    R2.copyTo(P2(cv::Rect(0, 0, 3, 3)));

    pose1.tvec.copyTo(P1(cv::Rect(3, 0, 1, 3)));
    pose2.tvec.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    cv::Mat triangulated;
    cv::triangulatePoints(P1, P2, vUnPt1, vUnPt2, triangulated);

    vpt3d.resize(vUnPt1.size());

    for (int i = 0, iend = int(vUnPt1.size()); i < iend; ++i)
    {

        float x = triangulated.at < float >(0, i);
        float y = triangulated.at < float >(1, i);
        float z = triangulated.at < float >(2, i);
        float w = triangulated.at < float >(3, i);

        vpt3d[i].x = x / w;
        vpt3d[i].y = y / w;
        vpt3d[i].z = z / w;
    }
}

#if DO_BA

/*!
@brief		BA for initial map
@param[in,out]		vPt3d		3d points
@param[in]			vDist1		points1
@param[in]			vDist2		points2
@param[in,out]		pose1		pose 1
@param[in,out]		pose2		pose 2
*/
bool CATAM::initialBA(
    std::vector<cv::Point3f> &vPt3d,
    const std::vector<cv::Point2f> &vDist1,
    const std::vector<cv::Point2f> &vDist2,
    sPose &pose1,
    sPose &pose2
)
{
    {
        double x1 = pose1.tvec.at<double>(0);
        double y1 = pose1.tvec.at<double>(1);
        double z1 = pose1.tvec.at<double>(2);

        double x2 = pose2.tvec.at<double>(0);
        double y2 = pose2.tvec.at<double>(1);
        double z2 = pose2.tvec.at<double>(2);

        double dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)); // + (z1 - z2) * (z1 - z2)
        printf("Initial distance %f \n", dist);
    }

    std::vector< std::vector<cv::Point2f> > imagePoints(2);
    imagePoints[0] = vDist1;
    imagePoints[1] = vDist2;

    std::vector< std::vector<int> > visibility(2);
    std::vector<int> vvis(vDist1.size(), 1);
    visibility[0] = vvis;
    visibility[1] = vvis;

    std::vector<cv::Mat> cameraMatrix(2);
    cameraMatrix[0] = mData.A.clone();
    cameraMatrix[1] = mData.A.clone();

    std::vector<cv::Mat> R(2);
    pose1.rvec.copyTo(R[0]);
    pose2.rvec.copyTo(R[1]);

    std::vector<cv::Mat> T(2);
    pose1.tvec.copyTo(T[0]);
    pose2.tvec.copyTo(T[1]);

    std::vector<cv::Mat> distCoeffs(2);
    distCoeffs[0] = cv::Mat::zeros(cv::Size(1, 5), CV_64F);
    distCoeffs[1] = cv::Mat::zeros(cv::Size(1, 5), CV_64F);

#if DO_BA
        LOGOUT("initial BA with %d points\n", int(vPt3d.size()));
        double val = -1;
        try
        {
            val = mBA.run(vPt3d, imagePoints, visibility, cameraMatrix, R, T, distCoeffs);
        }
        catch (cv::Exception& e)
        {
            return false;
        }
        LOGOUT("Initial error = \t %f\n", mBA.getInitialReprjError());
        LOGOUT("Final error = \t %f\n", mBA.getFinalReprjError());

        double reproError = mBA.getFinalReprjError();

        if (reproError > PARAMS.BA_MAX_ERROR)
        {
            return false;
        }

        if (val < 0.0)
        {
            return false;
        }

#endif

    R[0].copyTo(pose1.rvec);
    R[1].copyTo(pose2.rvec);

    T[0].copyTo(pose1.tvec);
    T[1].copyTo(pose2.tvec);

    double x1 = pose1.tvec.at<double>(0);
    double y1 = pose1.tvec.at<double>(1);
    double z1 = pose1.tvec.at<double>(2);

    double x2 = pose2.tvec.at<double>(0);
    double y2 = pose2.tvec.at<double>(1);
    double z2 = pose2.tvec.at<double>(2);

    double dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)); // + (z1 - z2) * (z1 - z2)
    printf("final distance %f \n", dist);

    return true;
}

#endif

/*!
@brief		mapping
@retval		new map is generated or not
*/
bool CATAM::makeMap(void)
{
    // select start and end points from new tracks
    std::vector<cv::Point2f> vStart;
    std::vector<cv::Point2f> vEnd;

    std::vector<sTrack*> vNewTrack;
    for (std::list<sTrack>::iterator it = mData.vTrack.begin(),
        itend = mData.vTrack.end();	it != itend; ++it)
    {
        if (it->ptID == NOID)
        {
            vStart.push_back(it->vPt[0]);
            vEnd.push_back(it->vPt.back());
            vNewTrack.push_back(&(*it));
        }
    }

    // if not enough points
    if (vStart.size() < PARAMS.MINPTS)
    {
        return false;
    }

    // undistort
    std::vector<cv::Point2f> vUndistStart, vUndistEnd;
    cv::undistortPoints(vStart, vUndistStart, mData.A, mData.D);
    cv::undistortPoints(vEnd, vUndistEnd, mData.A, mData.D);


    sKeyframe &lkf = mData.map.GetLastKeyframe();

    // compute initial pose from essential matrix for initialization
    if (mState == STATE::INIT)
    {
        //computePosefromE(vUndistStart, vUndistEnd, mPose.rvec, mPose.tvec);

        mPose.rvec = _rvecInit2;
        mPose.tvec = _tvecInit2;

        lkf.pose.rvec = _rvecInit1;
        lkf.pose.tvec = _tvecInit1;

        mData.haveScale = true;
        mData.scale = 1;
    }

    // triangulation
    std::vector<cv::Point3f> vPt3d;
    triangulate(vUndistStart, vUndistEnd, lkf.pose, mPose, vPt3d);

    // in initialization
    if (mState == STATE::INIT)
    {				
        // two view BA
        sPose tmpPose = lkf.pose;
        #if DO_BA

        /* for (auto it = vPt3d.begin(); it != vPt3d.end(); it++)
        {
            printf( it->x << " " << it->y << " " << it->z << "\n";
        }
        */

        if (!initialBA(vPt3d, vStart, vEnd, tmpPose, mPose))
        {
            return false;
        }
        else
        {

            lkf.pose = tmpPose;
        }
        #else
                
        lkf.pose = tmpPose;
                
        #endif
    }

    // check triangulation with reprojection error
    std::vector<cv::Point2f> vpPt1, vpPt2;
    cv::projectPoints(vPt3d, lkf.pose.rvec, lkf.pose.tvec, mData.A, mData.D, vpPt1);
    cv::projectPoints(vPt3d, mPose.rvec, mPose.tvec, mData.A, mData.D, vpPt2);

    int numinliers = 0;
    std::vector<bool> vInlier(vPt3d.size(), false);
    std::vector<sTrack*>::iterator it = vNewTrack.begin();

    std::vector<cv::Point3f> vinPt3d;
    std::vector<cv::KeyPoint> vinKpt;

    for (int i = 0, iend = int(vPt3d.size()); i < iend; ++i, ++it)
    {
        double dist1 = cv::norm(vStart[i] - vpPt1[i]);
        double dist2 = cv::norm(vEnd[i] - vpPt2[i]);

        if (dist1 < PARAMS.PROJERR && dist2 < PARAMS.PROJERR)
        {
            vInlier[i] = true;
            vinPt3d.push_back(vPt3d[i]);
            vinKpt.push_back((*it)->kpt);
            ++numinliers;
        }
    }

    // add points to keyframe and map
    cv::Mat vinDesc;
    if (vinKpt.size() != 0)
    {
        mDetector.Describe(lkf.img, vinKpt, vinDesc);
    }

    std::vector<int> vID;		// ID of points
    mData.map.UpdateLastKeyframe(vinPt3d, vinKpt, vinDesc, vID);

    // set point ID to new tracks
    it = vNewTrack.begin();
    int counter = 0;
    for (int i = 0, iend = int(vInlier.size()); i < iend; ++i, ++it)
    {
        if (vInlier[i])
        {
            (*it)->ptID = vID[counter];
            ++counter;
        }
    }

    return true;
}

/*!
brief		check criteria for adding a new keyframe
@retval		do mapping or not
*/
bool CATAM::mappingCriteria(void) const
{
    // middle point between current frame and nearest keyframe
    const sKeyframe &nkf = mData.map.GetNearestKeyframe(mPose);

    // compute camera location in world coordinate system
    cv::Mat R, nkfR;
    cv::Rodrigues(mPose.rvec, R);
    cv::Rodrigues(nkf.pose.rvec, nkfR);

    cv::Mat pos, mkfPos;
    pos = -R.inv() * mPose.tvec;
    mkfPos = -nkfR.inv() * nkf.pose.tvec;

    double distkeyframe = cv::norm(pos - mkfPos);
    cv::Point3f middle = (cv::Point3f(pos) + cv::Point3f(mkfPos)) / 2.0f;

    // select median of mapped points
    struct dist3D
    {
        double dist;
        int ID;
        bool operator< (const dist3D &r) const
        {
            return dist < r.dist;
        }
    };

    std::vector<dist3D> vDist3D;
    for (std::list<sTrack>::const_iterator it = mData.vTrack.begin(), itend = mData.vTrack.end();
        it != itend; ++it)
    {
        if (it->ptID != NOID)
        {
            dist3D tmp;
            tmp.ID = it->ptID;
            tmp.dist = cv::norm(mData.map.GetPoint(it->ptID) - middle);
            vDist3D.push_back(tmp);
        }
    }

    std::sort(vDist3D.begin(), vDist3D.end());
    const cv::Point3f &median = mData.map.GetPoint(vDist3D[int(vDist3D.size()) / 2].ID);

    double distpoints = cv::norm(median - middle);

    // mapping criteria
    if (PARAMS.BASETAN < distkeyframe / distpoints)
    {
        return true;
    }

    return false;
}

/*!
@brief		mapping
*/
void CATAM::mapping(void)
{
    makeMap();
    setKeyframe();

#ifndef MULTITHREAD
    BA();
#endif				
}

/*!
@brief		tracking and mapping
*/
int CATAM::trackAndMap(int &nbTrackedFeaturePoints)
{
#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    bool relocal = false;

    int mappedPts = trackFrame(nbTrackedFeaturePoints);		// track points

#if SLAM_PROFILING
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000.0f;
    printf( "  [trackAndMap] trackFrame : %f ms \n", duration_ms);
#endif
    if (mappedPts < PARAMS.MINPTS)
    {		// not enough mapped points
        relocal = true;					// start re-localization
        printf
        (
            "RELOCAL CASE 1  mapped points is %i while minimum is %i \n",
            mappedPts,
            PARAMS.MINPTS
        );
    }
    else
    {
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t22 = std::chrono::high_resolution_clock::now();
#endif
        bool is_pose_computed = computePose();
#if SLAM_PROFILING
        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
        duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t3 - t22).count()) / 1000.0f;
        printf("  [trackAndMap] computePose : %f ms \n", duration_ms);
        std::chrono::high_resolution_clock::time_point t33 = std::chrono::high_resolution_clock::now();
#endif
        if (!is_pose_computed)
        {
            printf("RELOCAL CASE 2 \n");
            relocal = true;
        }
        else
        {
#if SLAM_PROFILING
            std::chrono::high_resolution_clock::time_point t44 = std::chrono::high_resolution_clock::now();
            duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t44 - t33).count()) / 1000.0f;
            printf("  [trackAndMap] entering else took : %f ms \n", duration_ms);
#endif
            matchKeyframe();		// match with keyframe
#if SLAM_PROFILING
            std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
            duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000.0f;
            printf( "  [trackAndMap] matchKeyFrame : %f ms \n", duration_ms);
            std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();
#endif
            bool doMapping = mappingCriteria();
#if SLAM_PROFILING
            std::chrono::high_resolution_clock::time_point t8 = std::chrono::high_resolution_clock::now();
            auto duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count()) / 1000.0f;
            printf( "  [trackAndMap] mappingCriteria : %f ms \n", duration_ms);
#endif
#if DO_BA && !ONLY_DO_INITIAL_BA

            bool tmp = true;
            mBAMutex.lock();
            tmp = mDoingBA;
            mBAMutex.unlock();

            if (doMapping && !tmp) 
            {	// check mapping criteria
                mapping();			// mapping
            }

#else
            if (doMapping)
            {
                mapping();
#if SLAM_PROFILING
                std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
                duration_ms = (std::chrono::duration_cast<std::chrono::microseconds>(t5 - t8).count()) / 1000.0f;
                printf( "  [trackAndMap] mapping : %f ms \n", duration_ms);
#endif
            }
#endif
        }
    }

    if (relocal)
    {
       /* mData.clearAllTrack();
        LOGOUT("Lost\n");
        mText = "Go back to this view";
        mState = STATE::RELOCAL;*/
        printf("RESET because re-localization\n");
        reset();
    }

    return mappedPts;
}

/*!
@brief		initialization
*/
void CATAM::whileInitialize(void)
{
    if (mData.vPrevPt.size() > PARAMS.MINPTS)
    {	// if tracking points exist
        int nbTrackedFeaturePoints;
        int nbTrackedPoints = trackFrame(nbTrackedFeaturePoints);
        
        cv::Mat rvec, tvec;
        double reproError = 0;
        bool found = mCalibrator.EstimatePose(mGImg, mData.A, mData.D, rvec, tvec, reproError);

        int totalPoints = 0;
        int mappedPoints = 0;
        int newPoints = 0;

        getPointsCount(totalPoints, mappedPoints, newPoints);

        if (newPoints < PARAMS.RESET_BELOW_NB_POINTS)
        {
            reset();
            return;
        }

        if (!found)
        {
            return;
        }

        printf
        (
            "[2] Features points : %i tracked points : %i \n",
            newPoints,
            mappedPoints
        );

        if (found && nbTrackedFeaturePoints > PARAMS.MIN_FEATURES_POINT && reproError < PARAMS.MAX_REPRO_ERROR)
        {
            double x = tvec.at<double>(0);
            double y = tvec.at<double>(1);
            double z = tvec.at<double>(2);

            double dist = sqrt((x - _x) * (x - _x) + (y - _y) * (y - _y)); // + (z -_z) * (z - _z)

            //printf("dist is %f \n", dist);
            //printf("tracked points is %i with %i \n", nbTrackedPoints, nbTrackedFeaturePoints);
            if (dist > PARAMS.MIN_STEREO_DIST && reproError < PARAMS.MAX_REPRO_ERROR)
            {

                _rvecInit2 = rvec;
                _tvecInit2 = tvec;

                printf
                (
                    "2- Change state at %i, dist is %f, reprojection error %f\n",
                    mFrameNumber,
                    dist,
                    reproError
                );

                changeState(); // to startTAM();

            }
        }
    }
    else
    {
        LOGOUT("Initialization failed\n");
        reset();
    }
}

/*!
@brief		local bundle adjustment
*/
#if DO_BA && !ONLY_DO_INITIAL_BA

void CATAM::BA(void)
{
    while (1)
    {

        STATE mainState = mState;

        if (mainState == STATE::TAM)
        {

            // get data from map
            sBAData baData;
            bool copied = mData.map.CopytoBA(baData);
            std::vector<sKeyframe> &vKf = baData.vKeyframe;
            int numKeyframes = int(vKf.size());

            if (copied && numKeyframes > 2)
            {

                mBAMutex.lock();
                mDoingBA = true;
                mBAMutex.unlock();

                // check visibility of mapped points in each keyframe
                std::vector<cv::Point3f> &vPt3d = baData.vPt3d;
                std::vector<int> checkVis(vPt3d.size(), 0);

                for (int i = 0, iend = numKeyframes; i < iend; ++i)
                {
                    sKeyframe &kf = vKf[i];
                    for (int j = 0, jend = int(kf.vPtID.size()); j < jend; ++j)
                    {
                        ++checkVis[kf.vPtID[j]];
                    }
                }

                // select visible mapped points
                std::vector<cv::Point3f> vUsedPt3d;
                std::vector<int> vVisibleID;

                for (int i = 0, iend = int(checkVis.size()); i < iend; ++i)
                {
                    if (checkVis[i] > 1)
                    {	// should be visible in more than two views
                        int num = int(vUsedPt3d.size());
                        checkVis[i] = num;
                        vVisibleID.push_back(i);
                        vUsedPt3d.push_back(vPt3d[i]);
                    }
                    else
                    {
                        checkVis[i] = -1;
                    }
                }

                // set data for cvsba
                std::vector< std::vector<cv::Point2f> > imagePoints(numKeyframes);
                std::vector< std::vector<int> > visibility(numKeyframes);
                std::vector<cv::Mat> cameraMatrix(numKeyframes);
                std::vector<cv::Mat> R(numKeyframes);
                std::vector<cv::Mat> T(numKeyframes);
                std::vector<cv::Mat> distCoeffs(numKeyframes);

                for (int i = 0; i < numKeyframes; ++i)
                {

                    std::vector<cv::Point2f> points(vUsedPt3d.size());
                    std::vector<int> vis(vUsedPt3d.size(), 0);

                    sKeyframe &kf = vKf[i];
                    for (int j = 0, jend = int(kf.vPtID.size()); j < jend; ++j)
                    {
                        int id = checkVis[kf.vPtID[j]];

                        if (id != -1)
                        {
                            points[id] = kf.vPt[j];
                            vis[id] = 1;
                        }
                    }

                    imagePoints[i] = points;
                    visibility[i] = vis;
                    cameraMatrix[i] = mData.A;
                    distCoeffs[i] = mData.D;
                    kf.pose.rvec.copyTo(R[i]);
                    kf.pose.tvec.copyTo(T[i]);
                }

                LOGOUT("BA started with %d points %d frames\n", int(vPt3d.size()), numKeyframes);
                double val = -1;
                try
                {
                    val = mBA.run(vUsedPt3d, imagePoints, visibility, cameraMatrix, R, T, distCoeffs);
                }
                catch (cv::Exception& e)
                {
                    _needReset = true;
                }

                if (val < 0)
                {
                    LOGOUT("BA failed\n");
                }
                else
                {
                    LOGOUT("Initial error = \t %f\n", mBA.getInitialReprjError());

                    LOGOUT("Final error = \t %f\n", mBA.getFinalReprjError());

                    double reproError = mBA.getFinalReprjError();

                    if (reproError > PARAMS.BA_MAX_ERROR)
                    {
                        _needReset = true;
                    }

                    for (int i = 0; i < numKeyframes; ++i)
                    {
                        sKeyframe &kf = vKf[i];

                        R[i].copyTo(kf.pose.rvec);
                        T[i].copyTo(kf.pose.tvec);
                    }

                    for (int i = 0, iend = int(vVisibleID.size()); i < iend; ++i)
                    {
                        int id = vVisibleID[i];
                        vPt3d[id] = vUsedPt3d[i];
                    }

                    baData.vVisibleID = vVisibleID;
                    mData.map.CopyfromBA(baData);
                }

                mBAMutex.lock();
                mDoingBA = false;
                mBAMutex.unlock();
            }
        }
        else if (mainState == STATE::CLOSE)
        {
            break;
        }

#ifndef MULTITHREAD
        break;
#endif
    }
}
#endif



/*!
@brief		transform local coordinate to world coordinate
@param[in]	local		local coordinate
@param[out]	world		world coordinate
*/
void CATAM::transformToWorld(const sPose &local, sPose &world) const
{
    cv::Mat tmpM;
    local.getM(tmpM);

    tmpM *= mData.scale;

    cv::Mat M = tmpM * mData.transMat;

    cv::Rodrigues(M(cv::Rect(0, 0, 3, 3)), world.rvec);
    M(cv::Rect(3, 0, 1, 3)).copyTo(world.tvec);
}

/*!
@brief		get pose in world coordinate system from a calibration pattern
@param[out]	pose	pose
*/
bool CATAM::getWorldCoordinate(sPose &pose) const
{
    double reproError = 0;
    bool found = mCalibrator.EstimatePose(mGImg, mData.A, mData.D, pose.rvec, pose.tvec, reproError);

    if (!found)
    {
        LOGOUT("Calibration board not found\n");
    }
    else
    {
        printf("reprojection error is %f\n", reproError);
    }

    return found;
}

/*!
@brief		registration with world coordinate system
*/
void CATAM::registerWorld(void)
{
    printf("Registering world at %i\n", mFrameNumber);
    // get pair of world and local pose
    sPose world;
    if (getWorldCoordinate(world))
    {
        std::pair<sPose, sPose> tmp;
        tmp.first = world;		// world pose
        tmp.second = mPose;		// local pose
        mData.vPosePair.push_back(tmp);
        mText = "Translate camera and capture again";
    }

    if (mData.vPosePair.size() > 1)
    {
        std::vector<double> vScale;

        // compute scale from all pairs
        for (int i = 0, iend = int(mData.vPosePair.size()) - 1; i < iend; ++i)
        {
            // compute distance between two frames and compute ratio
            for (int j = i + 1, jend = int(mData.vPosePair.size()); j < jend; ++j)
            {
                cv::Mat Ri, Rj;
                mData.vPosePair[i].second.getR(Ri);
                mData.vPosePair[j].second.getR(Rj);

                cv::Mat R = Ri * Rj.inv();
                double numerator = cv::norm(mData.vPosePair[i].first.tvec - R * mData.vPosePair[j].first.tvec);
                double denominator = cv::norm(mData.vPosePair[i].second.tvec - R * mData.vPosePair[j].second.tvec);

                vScale.push_back(numerator / denominator);
            }
        }

        // average scale
        mData.scale = std::accumulate(vScale.begin(), vScale.end(), 0.0) / double(vScale.size());

        // compute transformation matrix from local to world after scaling from pair of median scale
        cv::Mat transMat = cv::Mat::zeros(cv::Size(4, 4), CV_64F);
        int size = int(mData.vPosePair.size());
        for (int i = 0; i < size; ++i)
        {
            sPose world = mData.vPosePair[i].first;
            sPose local = mData.vPosePair[i].second;
            local.tvec *= mData.scale;

            cv::Mat worldM, localM;
            world.getM(worldM);
            local.getM(localM);

            cv::add(transMat.clone(), localM.inv() * worldM, transMat);
        }

        // average rotation matrix
        cv::Mat X, D, Y;
        cv::SVD::compute(transMat(cv::Rect(0, 0, 3, 3)), D, X, Y);
        mData.transMat(cv::Rect(0, 0, 3, 3)) = (X * Y) / mData.scale;

        // average translation vector
        mData.transMat(cv::Rect(3, 0, 1, 3)) = transMat(cv::Rect(3, 0, 1, 3)) / double(size) / mData.scale;

        mData.haveScale = true;

        printf("Average scale factor is %f\n", mData.scale);

        mText = "Capture more for improvement";
    }
}


/*!
@brief		change re-localization view
*/
void CATAM::changeRelocalImage(void)
{
    mData.map.GetRandomKeyFramePose(mPose);
}

/*!
@brief		re-localization
*/
void CATAM::relocalize(void)
{
    if (mData.vKpt.size() == 0)
    {
        return;
    }
    else if (matchKeyframe())
    {
        if (mData.haveScale)
        {
            mText = "Re-localized";
        }
        else
        {
            mText = "Capture calibration board and press space";
        }

        LOGOUT("Re-localized\n");
        mState = STATE::TAM;
    }
}

/*!
@brief			overlay view
@param[in,out]	img		color image
*/
void CATAM::drawView(cv::Mat &img)
{
    const sKeyframe& kf = mData.map.GetNearestKeyframe(mPose);

    cv::Mat mono = cv::Mat(img.size(), CV_8U);
    cv::Canny(kf.img, mono, 10, 100);

    std::vector<cv::Mat> color(3, mono);

    cv::Mat cimg;
    cv::merge(color, cimg);

    double alpha = 0.5;
    double beta = (1.0 - alpha);
    addWeighted(img.clone(), alpha, cimg, beta, 0.0, img);
}

/*!
@brief		load challenge points
@param[in]	file name
*/
void CATAM::loadChallenge(const std::string &name)
{
    std::ifstream in(name);

    if (in.is_open())
    {

        int num;
        in >> num;

        for (int i = 0; i < num; ++i)
        {
            if (!in.eof())
            {
                int id;
                cv::Point3f pt;
                in >> id >> pt.x >> pt.y >> pt.z;
                mData.vChallenge.insert(std::pair<int, cv::Point3f>(id, pt));
            }
        }
    }
    else
    {
        LOGOUT("%s not found\n", name.c_str());

        // to show origin of world coordinate system
        mData.vChallenge.insert(std::pair<int, cv::Point3f>(0, cv::Point3f(0, 0, 0)));
    }
}

/*!
@brief				draw challenge coordinates
@param[in,out]		img		color image
*/
void CATAM::drawChallenge(cv::Mat &img)
{
    if (mData.haveScale)
    {

        cv::Scalar col(0, 0, 255);

        const int lineWidth = 2;
        const int radius = 2;

        // world pose computation
        sPose tmp;
        transformToWorld(mPose, tmp);

        std::vector<cv::Point2f> vPt2d;
        std::vector<cv::Point3f> vPt3d;
        int ID = 0;

        // draw targets defined in world coordinate system
        int i = 0;
        for (std::map<int, cv::Point3f>::const_iterator it = mData.vChallenge.begin(),
            itend = mData.vChallenge.end(); it != itend; ++it, ++i)
        {
            if (mChallengeNumber == i)
            {
                ID = it->first;
                vPt3d.push_back(it->second);
                break;
            }
        }

        if (vPt3d.size() == 1)
        {
            cv::projectPoints(vPt3d, tmp.rvec, tmp.tvec, mData.A, mData.D, vPt2d);

            cv::Point pt = vPt2d[0];
            cv::circle(img, pt, radius, col, lineWidth);
            cv::putText(img, std::to_string(ID), pt + cv::Point(radius, 0), cv::FONT_HERSHEY_SIMPLEX, 1.5, col, 2);
        }
        else
        {
            mText = "No more challenge points";
        }
    }
}

/*!
@brief			draw button
@param[out]		img		color image
*/
void CATAM::drawButton(cv::Mat &img)
{
    for (int i = 0, iend = int(mvButton.size()); i < iend; ++i)
    {	// for each button
        cv::Scalar col(0, 255, 0);
        int thickness = 1;

        // show buttons
        cv::rectangle(img, mvButton[i].r, col, thickness);

        // show text on button
        cv::putText
        (
            img, 
            mvButton[i].name, 
            cv::Point(mvButton[i].r.x, mvButton[i].r.y + mvButton[i].r.height / 2), 
            cv::FONT_HERSHEY_SIMPLEX, 
            0.5, 
            col, 
            thickness
        );
    }
}

/*!
@brief		check button is pressed or not
@param[in]	x		x coordinate
@param[in]	y		y coordinate
*/
void CATAM::checkButton(const int x, const int y)
{
    for (int i = 0, iend = int(mvButton.size()); i < iend; ++i)
    {		// for each button
        if (mvButton[i].in(x, y))
        {		// check inside button or not
            mMouse = mvButton[i].key;
            break;
        }
    }
}

/*!
@brief		generate button
*/
void CATAM::generateButton(void)
{
    // define button size
    const int ratio = 13;
    const int w = mImg.cols / ratio;
    const int h = mImg.rows / ratio;

    button tmp;

    // space
    tmp.set(mImg.cols - w, mImg.rows - h, w, h);
    tmp.key = ' ';
    tmp.name = "space";
    mvButton.push_back(tmp);

    // c
    tmp.set(mImg.cols - w, mImg.rows - 2 * h, w, h);
    tmp.key = 'c';
    tmp.name = "  c  ";
    mvButton.push_back(tmp);

    // n
    tmp.set(mImg.cols - w, mImg.rows - 3 * h, w, h);
    tmp.key = 'n';
    tmp.name = "  n  ";
    mvButton.push_back(tmp);

    // r
    tmp.set(mImg.cols - w, mImg.rows - 4 * h, w, h);
    tmp.key = 'r';
    tmp.name = "  r  ";
    mvButton.push_back(tmp);

    // q
    tmp.set(mImg.cols - w, mImg.rows - 5 * h, w, h);
    tmp.key = 'q';
    tmp.name = " Quit";
    mvButton.push_back(tmp);
}


/*!
@brief		OpenCV mouse operation
*/
void CATAM::mouse(int event, int x, int y, int flags)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        checkButton(x, y);
        break;
    default:
        break;
    }
}

/*!
@brief		mouse dummy for OpenCV mouse callback
*/
void CATAM::mousedummy(int event, int x, int y, int flags, void* param)
{
    CATAM *patam = reinterpret_cast<CATAM*>(param);
    patam->mouse(event, x, y, flags);
}
/*!
@brief		user's operation
@param[in]	input key
@retval		ESC is pressed or not
*/
bool CATAM::operation(const int key)
{
    if (key == ' ' || mMouse == ' ')
    {		// change state
        changeState();
    }
    else if (key == 'r' || mMouse == 'r')
    {	// reset
        reset();
    }
    else if (key == 'n' || mMouse == 'n')
    {
        if (mState == STATE::RELOCAL)
        {		// change image for re-localization
            changeRelocalImage();
        }
        else if (mState == STATE::TAM)
        {		// manual mapping
            mapping();
        }
    }
    else if (key == 'c' || mMouse == 'c')
    {
        ++mChallengeNumber;
    }
    else if (key == 'q' || mMouse == 'q')
    {	// exit
        return true;
    }

    mMouse = -1;	// clear mouse data

    return false;
}

/*!
@brief			draw on images
@param[in,out]	img		color image
*/
void CATAM::draw(cv::Mat &img)
{
    // each process
    switch (mState)
    {
    case STATE::INIT:
        drawTrack(img);
        break;
    case STATE::TAM:
        drawTrack(img);
        drawMap(img);
        drawGrid(img);
        drawChallenge(img);
        break;
    case STATE::RELOCAL:
        drawView(img);
        break;
    default:
        break;
    }

    drawProcess(img);
    drawButton(img);
}

/*!
@brief			draw process
@param[in,out]	img		color image
*/
void CATAM::drawProcess(cv::Mat &img) const
{
    // draw process at the bottom
    cv::Scalar textCol(0, 255, 0);
    cv::putText(img, mText, cv::Point(0, img.rows - 5), cv::FONT_HERSHEY_SIMPLEX, 0.8, textCol, 2);

    // draw fps	at the top
    textCol = cv::Scalar(0, 255, 0);
    cv::putText(img, std::to_string(int(mFPS)) + " FPS", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, textCol, 1);

    // draw doing BA or not
#if DO_BA
    if (mDoingBA)
    {
        textCol = cv::Scalar(0, 0, 255);
        cv::putText(img, "Doing BA", cv::Point(img.cols - 75, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, textCol, 1);
    }
#endif
}

/*!
@brief		draw map
@param[in]	img		color image
*/
void CATAM::drawMap(cv::Mat &img) const
{
    const std::list<sTrack> &vTrack = mData.vTrack;

    std::vector<cv::Point3f> vPt3d;

    // select mapped tracks
    for (std::list<sTrack>::const_iterator it = vTrack.begin(), itend = vTrack.end();
        it != itend; ++it)
    {
        if (it->ptID != NOID)
        {
            vPt3d.push_back(mData.map.GetPoint(it->ptID));
        }
    }

    // project mapped tracks
    if (vPt3d.size() != 0)
    {
        std::vector<cv::Point2f> vpt2d;
        cv::projectPoints(vPt3d, mPose.rvec, mPose.tvec, mData.A, mData.D, vpt2d);

        cv::Scalar col(0, 0, 255);		// map color
        const int size = 1;
        for (int i = 0, iend = int(vpt2d.size()); i < iend; ++i)
        {
            cv::circle(img, cv::Point(vpt2d[i]), size, col, -1);
        }
    }
}

/*!
@brief		draw grid on calibration board
@param[in]	img		color image
*/
void CATAM::drawGrid(cv::Mat &img) const
{
    if (mData.haveScale)
    {	// scale already estimated

        sPose tmp;
        transformToWorld(mPose, tmp);

        // line drawing
        cv::Scalar col(0, 255, 0);
        const int lineWidth = 2;

        const int size = 3;		// grid size
        std::vector<cv::Point3f> vPt3d(2 * size);		// for two sides
        const float interval = CHESSBOARD_SIZE;
        //const float interval = 3 * CHESSBOARD_SIZE * 10;

        // y axis
        for (int i = 0; i < size; ++i)
        {
            vPt3d[2 * i] = cv::Point3f(interval*i, 0, 0);
            vPt3d[2 * i + 1] = cv::Point3f(interval*i, interval*(size - 1), 0);
        }

        std::vector<cv::Point2f> vPt2d;
        cv::projectPoints(vPt3d, tmp.rvec, tmp.tvec, mData.A, mData.D, vPt2d);

        for (int i = 0, iend = int(vPt2d.size()) / 2; i < iend; ++i)
        {
            cv::line(img, cv::Point(vPt2d[2 * i]), cv::Point(vPt2d[2 * i + 1]), col, lineWidth);
        }

        // x axis
        for (int i = 0; i < size; ++i)
        {
            vPt3d[2 * i] = cv::Point3f(0, interval*i, 0);
            vPt3d[2 * i + 1] = cv::Point3f(interval*(size - 1), interval*i, 0);
        }

        cv::projectPoints(vPt3d, tmp.rvec, tmp.tvec, mData.A, mData.D, vPt2d);

        for (int i = 0, iend = int(vPt2d.size()) / 2; i < iend; ++i)
        {
            cv::line(img, cv::Point(vPt2d[2 * i]), cv::Point(vPt2d[2 * i + 1]), col, lineWidth);
        }
    }
}

/*!
@brief		draw tracked points
@param[in]	img		image
*/
void CATAM::drawTrack(cv::Mat &img) const
{
    const int pointSize = 2;

    const std::list<sTrack> &vTrack = mData.vTrack;

    cv::Scalar border(0, 0, 0);			// circle border
    cv::Scalar mapped(255, 255, 255);	// mapped
    cv::Scalar newpt(255, 0, 0);		// not mapped

    for (std::list<sTrack>::const_iterator it = vTrack.begin(), itend = vTrack.end();
        it != itend; ++it)
    {
        if (it->ptID != NOID)
        {
            //cv::circle(img, cv::Point(it->vPt.back()), pointSize, border, -1);
            cv::circle(img, cv::Point(it->vPt.back()), pointSize - 1, mapped, -1);
        }
        else
        {
            //cv::circle(img, cv::Point(it->vPt.back()), pointSize, border, -1);
            cv::circle(img, cv::Point(it->vPt.back()), pointSize - 1, newpt, -1);
        }
    }
}

void CATAM::getPointsCount(int &totalPoints, int &mappedPoints, int &newPoints) const
{
    const std::list<sTrack> &vTrack = mData.vTrack;

    for (std::list<sTrack>::const_iterator it = vTrack.begin(), itend = vTrack.end();
        it != itend; ++it)
    {
        if (it->ptID != NOID)
        {
            mappedPoints++;
        }
        else
        {
            newPoints++;
        }
        totalPoints++;
    }
}