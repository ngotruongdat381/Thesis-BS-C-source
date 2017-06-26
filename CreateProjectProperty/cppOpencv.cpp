#pragma once

#include "cppOpencv.h"

//using namespace std;
//using namespace cv;

double MYcppGui::OverlapPercentage(vector<Point2f> groundTruth, vector<Point2f> points) {
	double NUM_OVERLAP_POINT = 0;
	Mat TestMat = Mat::zeros(userInputFrame.size(), CV_8UC3);
	TestMat = Scalar::all(0);
	for (int i = 0; i < points.size() - 1; i++) {
		line(TestMat, points[i], points[i + 1], white, 20, 8, 0);
	}
	for (int i = 0; i < groundTruth.size(); i++) {
		Vec3b color = TestMat.at<Vec3b>(groundTruth[i]);
		Scalar cvColor = Scalar(color[0], color[1], color[2]);
		circle(TestMat, groundTruth[i], 0.5, yellow, -1, 8);
		if (cvColor == white) {
			NUM_OVERLAP_POINT++;
			circle(TestMat, groundTruth[i], 0.5, green, -1, 8);
		}
	}
	cout << NUM_OVERLAP_POINT << "/" << groundTruth.size() << " : " << NUM_OVERLAP_POINT/groundTruth.size() << endl;
	return NUM_OVERLAP_POINT / groundTruth.size();
}

vector<double> MYcppGui::CompareToGroundTruth(vector<vector<Point2f>> line) {
	vector<double> result;
	for (int k = 0; k < 2; k++) {
		bool firstPoint, lastPoint;
		if (line[k][0] == userInput[k][0]) {
			lastPoint = true;
		}
		if (line[k].back() == userInput[k].back()) {
			firstPoint = true;
		}
		double percentage = OverlapPercentage(userInput[k], line[k]);
		result.push_back(percentage);
	}
	return result;
}

bool CheckCommon(std::vector<Point2f> inVectorA, std::vector<Point2f> nVectorB)
{
	return std::find_first_of(inVectorA.begin(), inVectorA.end(),
		nVectorB.begin(), nVectorB.end()) != inVectorA.end();
}

Mat Combine2MatSideBySide(Mat &im1, Mat &im2)
{
	Size sz1 = im1.size();
	Size sz2 = im2.size();
	Mat im3(sz1.height, sz1.width + sz2.width, CV_8UC3);
	Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
	im1.copyTo(left);
	Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
	im2.copyTo(right);
	return im3;
}

string GetTime() {
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%d-%m-%Y %I:%M:%S", timeinfo);
	std::string str(buffer);
	return str;
}

Mat ChangeBrightness(Mat& image, int beta = 50) {
	double alpha = 1.0; /*< Simple contrast control */
	//int beta = 50;       /*< Simple brightness control */
	Mat new_image = Mat::zeros(image.size(), image.type());

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			for (int c = 0; c < 3; c++) {
				new_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(image.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
	return new_image;
}

bool R1(int R, int G, int B) {
	bool e1 = (R>95) && (G>40) && (B>20) && ((max(R, max(G, B)) - min(R, min(G, B)))>15) && (abs(R - G)>15) && (R>G) && (R>B);
	bool e2 = (R>220) && (G>210) && (B>170) && (abs(R - G) <= 15) && (R>B) && (G>B);
	return (e1 || e2);
}

bool R2(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862*Cb + 20;
	bool e4 = Cr >= 0.3448*Cb + 76.2069;
	bool e5 = Cr >= -4.5652*Cb + 234.5652;
	bool e6 = Cr <= -1.15*Cb + 301.75;
	bool e7 = Cr <= -2.2857*Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
	return (H<25) || (H > 230);
}

Mat GetSkin(Mat const &src) {
	// allocate the result matrix
	Mat dst = src.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);

	Mat src_ycrcb, src_hsv;
	// OpenCV scales the YCrCb components, so that they
	// cover the whole value range of [0,255], so there's
	// no need to scale the values:
	cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
	// OpenCV scales the Hue Channel to [0,180] for
	// 8bit images, so make sure we are operating on
	// the full spectrum from [0,360] by using floating
	// point precision:
	src.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
	// Now scale the values between [0,255]:
	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			// apply rgb rule
			bool a = R1(R, G, B);

			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			// apply ycrcb rule
			bool b = R2(Y, Cr, Cb);

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			// apply hsv rule
			bool c = R3(H, S, V);

			if (!(a&&b&&c))
				dst.ptr<Vec3b>(i)[j] = cblack;
		}
	}
	return dst;
}


void overlay_whiteBackground_image() {
	//Add mustache
	//Load our overlay image : mustache.png
	/*Mat imgMustache = imread("D:\\605\\Source code\\dataset\\Sticker\\mustache.jpg", IMREAD_UNCHANGED);
	double mustacheHeight = shapes_face[0].part(51).y() - shapes_face[0].part(33).y();
	double mustacWidth = mustacheHeight / imgMustache.size().height * imgMustache.size().width;*/

	//Mat mask1, src1;
	//resize(imgMustache, mask1, Size(mustacWidth, mustacheHeight));

	//// ROI selection
	//Rect roi(shapes_face[0].part(33).x() - mustacWidth / 2, shapes_face[0].part(33).y(), mustacWidth, mustacheHeight);
	//face_detection_frame(roi).copyTo(src1);

	//// to make the white region transparent
	//Mat mask2, m, m1;
	//cvtColor(mask1, mask2, CV_BGR2GRAY);
	//threshold(mask2, mask2, 230, 255, CV_THRESH_BINARY_INV);

	//vector<Mat> maskChannels(3), result_mask(3);
	//split(mask1, maskChannels);
	//bitwise_and(maskChannels[0], mask2, result_mask[0]);
	//bitwise_and(maskChannels[1], mask2, result_mask[1]);
	//bitwise_and(maskChannels[2], mask2, result_mask[2]);
	//merge(result_mask, m);         //    imshow("m",m);

	//mask2 = 255 - mask2;
	//vector<Mat> srcChannels(3);
	//split(src1, srcChannels);
	//bitwise_and(srcChannels[0], mask2, result_mask[0]);
	//bitwise_and(srcChannels[1], mask2, result_mask[1]);
	//bitwise_and(srcChannels[2], mask2, result_mask[2]);
	//merge(result_mask, m1);        //    imshow("m1",m1);

	//addWeighted(m, 1, m1, 1, 0, m1);    //    imshow("m2",m1);

	//m1.copyTo(face_detection_frame(roi));
}

Mat overlrayImage(Mat background, Mat foreground) {

	Mat dst = background.clone();
	Mat resizedFG;
	resize(foreground, resizedFG, background.size());


	for (int y = 0; y < (int)(background.rows); ++y) {
		for (int x = 0; x < (int)(background.cols); ++x) {
			Vec3b b = background.at<cv::Vec3b>(y, x);
			Vec4b& f = resizedFG.at<Vec4b>(y, x);

			double alpha = f[3] / 255.0;

			Vec3b d;
			for (int k = 0; k < 3; ++k) {
				d[k] = f[k] * alpha + b[k] * (1.0 - alpha);
			}

			dst.at<cv::Vec3b>(y, x) = d;
		}
	}
	return dst;
}

void overlayMask(Mat &background, Mat &foreground, Mat& mask) {
	Mat m, m1;
	vector<Mat> maskChannels(3), result_mask(3);
	split(foreground, maskChannels);
	bitwise_and(maskChannels[0], mask, result_mask[0]);
	bitwise_and(maskChannels[1], mask, result_mask[1]);
	bitwise_and(maskChannels[2], mask, result_mask[2]);
	merge(result_mask, m);         //    imshow("m",m);

	mask = 255 - mask;
	vector<Mat> srcChannels(3);
	split(background, srcChannels);
	bitwise_and(srcChannels[0], mask, result_mask[0]);
	bitwise_and(srcChannels[1], mask, result_mask[1]);
	bitwise_and(srcChannels[2], mask, result_mask[2]);
	merge(result_mask, m1);            imshow("m1", m1);

	addWeighted(m, 1, m1, 1, 0, m1);        imshow("m2", m1);

	m1.copyTo(background);
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
bool onSegment(Point2f p, Point2f q, Point2f r)
{
	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
		return true;

	return false;
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(Point2f p, Point2f q, Point2f r)
{
	// See http://www.geeksforgeeks.org/orientation-3-ordered-points/
	// for details of below formula.
	int val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0;  // colinear

	return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool doIntersect(Point2f p1, Point2f q1, Point2f p2, Point2f q2)
{
	// Find the four orientations needed for general and
	// special cases
	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	// General case
	if (o1 != o2 && o3 != o4)
		return true;

	// Special Cases
	// p1, q1 and p2 are colinear and p2 lies on segment p1q1
	if (o1 == 0 && onSegment(p1, p2, q1)) return true;

	// p1, q1 and p2 are colinear and q2 lies on segment p1q1
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;

	// p2, q2 and p1 are colinear and p1 lies on segment p2q2
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;

	// p2, q2 and q1 are colinear and q1 lies on segment p2q2
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false; // Doesn't fall in any of the above cases
}

MYcppGui::MYcppGui()
{
	checking_block = 0;
	dlib::deserialize("D:\\shape_predictor_68_face_landmarks.dat") >> shape_predictor;
}

MYcppGui::~MYcppGui()
{
	cvDestroyAllWindows();
}

vector<vector<Point2f>> SimplifizeResult(vector<vector<Point2f>> result) {
	vector<vector<Point2f>> Simplifized;
	Simplifized.resize(result.size());
	for (int k = 0; k < 2; k++) {
		approxPolyDP(Mat(result[k]), Simplifized[k], 10, true);
	}
	return Simplifized;
}

vector<vector<Point2f>> MYcppGui::readUserInput(string path)
{
	vector<cv::Point2f> leftShoulderInput;
	vector<cv::Point2f> rightShoulderInput;
	vector<vector<Point2f> > inputs;

	std::ifstream infile(path);
	std::string line;
	double previous_a = 0;
	bool left = true;

	if (infile.good()){
		while (std::getline(infile, line))
		{
			std::istringstream iss(line);
			double a, b;
			if (!(iss >> a >> b)) { break; } // error

			//Move to right shoulder input
			if (abs(a - previous_a) > 50 && previous_a != 0) {
				left = false;
			}

			if (left) {
				leftShoulderInput.push_back(Point2f(a, b));
			}
			else {
				rightShoulderInput.push_back(Point2f(a, b));
			}
			previous_a = a;
		}
		//Make the input follow this way: --->  <---
		reverse(leftShoulderInput.begin(), leftShoulderInput.end());
		reverse(rightShoulderInput.begin(), rightShoulderInput.end());

		inputs.push_back(leftShoulderInput);
		inputs.push_back(rightShoulderInput);
	}	

	return inputs;
}

void MYcppGui::AddUserInput(string path)
{
	//if (TRACKING_MODE) {
		userInput = readUserInput(path);
		current_shoulderLine = vector<vector<Point2f>>(userInput);
		
		//Simplifized User Input.
		simplifizedUserInput = SimplifizeResult(userInput);
		simplifized_current_shoulderLine = vector<vector<Point2f>>(simplifizedUserInput);
		userInputFrame = NULL;
	//}

}

void MYcppGui::GetSticker(string name, bool changeDirection) {
	for (int i = 0; i < 100; i++) {
		string path = "D:\\605\\Source code\\dataset\\Sticker\\" + name + "\\" + to_string(i) + ".png";
		Mat image = imread(path, -1);
		if (!image.data)                              // Check for invalid input
		{
			cout << "Read sticker: DONE!" << std::endl;
			break;
		}
		if (changeDirection) {
			cv::Mat dst;               // dst must be a different Mat
			cv::flip(image, dst, 1);     // because you can't flip in-place (leads to segfault)
			image = dst.clone();
		}
		stickerFrames.push_back(image);
	}
}

int MYcppGui::myCppLoadAndShowRGB(string fileName)
{
	cout << fileName << endl;
	// my OpenCV C++ codes uses only cxcore, highgui and imgproc 
	fileName = "D:\\train_02.jpg";
	cv::Mat img_input = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
	cv::namedWindow("Source", CV_WINDOW_NORMAL);
	cv::resizeWindow("Source", 282, 502);
	cv::imshow("Source", img_input);
	return 0;
}

void MYcppGui::VideoProcessing(string fileName) {
	VideoCapture capture(fileName);
	if (!capture.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return;
	}
	double fps = capture.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
	bool skip_frame = false;
	if (fps > 35) {
		skip_frame = true;
	}

	cout << "Frame per seconds : " << fps << endl;

	cv::namedWindow("Source", CV_WINDOW_NORMAL);
	cv::resizeWindow("Source", 530, 700);

	//Save the vide0
	int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	
	int str_lenght = fileName.size();
	string noFile = fileName.substr(str_lenght - 6, str_lenght - 5); //dont know why it crop from str_lenght - 6 to the end, so I crop it once again
	noFile = noFile.substr(0, 2);

	VideoWriter video("output_" + noFile + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), true);

	//----------------------Sticker---------------------------
	GetSticker(stickerName, false);

	//-------------------------collect sample color of shouder--------------------
	if (TRACKING_MODE) {
		collectColorShoulderFromInput();
	}
	
	while (1)
	{
		//skip frame if the video in 60fps
		if (skip_frame && nth % 2 == 0)
		{
			nth++;
			continue;
		}
		cout << nth << " th frame" << endl;
		Mat frame, face_processed;
		bool bSuccess = capture.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			capture = VideoCapture(fileName);
			bSuccess = capture.read(frame);
		}
		face_processed = frame.clone();
		ImageProcessing_Final(face_processed, false, false, true);

		if (TEST_MODE) {
			frame = face_processed;
		}

		if (STICKER_MODE) {
			AddSticker(frame);
		}
		
		cv::imshow("Source", frame);
		video.write(frame);
		
		nth++;
		index_stickerFrames++;
		index_stickerFrames %= stickerFrames.size();

		if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
		
	}
}


vector<Mat> MYcppGui::ImageProcessing_Final(Mat &frame, bool withUserInput, bool isTesting, bool DebugLine) {

	vector<Mat> returnMats;
	Mat src, face_detection_frame; // change name soon
	src = frame.clone(); // Use for Canny
	face_detection_frame = frame.clone(); // Use for shoulder detection

	if (userInputFrame.empty())
	{
		userInputFrame = frame.clone();
	}

	Mat detected_edges;
	if (isTesting) {
		cv::namedWindow("Erosion After Canny", CV_WINDOW_NORMAL);
		cv::resizeWindow("Erosion After Canny", 282, 502);
		cv::namedWindow("Canny Only", CV_WINDOW_NORMAL);
		cv::resizeWindow("Canny Only", 282, 502);
		//--------------------------------blur ----------------------------
		int blurIndex = 7;
		medianBlur(frame, frame, blurIndex);

		//--------------------------------Morphology Open Close ----------------------------
		Morphology_Operations(frame);

		//----------------------------------Canny ---------------------
		CannyProcessing(frame, detected_edges);

		//----------------------------- Erosion after canny --------------------
		int erosion_size = 6;

		Mat element = getStructuringElement(cv::MORPH_CROSS,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
		cv::dilate(detected_edges, detected_edges, element);

		cv::imshow("Erosion After Canny", detected_edges);
	}

	//------------- Face Detection ------------------
	std::vector<dlib::full_object_detection> shapes_face;
	shapes_face = face_detection_update(face_detection_frame);

	//No face is detected
	if (shapes_face.size() == 0) {
		return returnMats;
	}

	//testing
	for (int i = 0; i < 5; i++) {
		line(face_detection_frame, Point2f(shapes_face[0].part(i).x(), shapes_face[0].part(i).y()),
			Point2f(shapes_face[0].part(i + 1).x(), shapes_face[0].part(i + 1).y()), green, 5, 8, 0);
		line(face_detection_frame, Point2f(shapes_face[0].part(16-i).x(), shapes_face[0].part(16-i).y()),
			Point2f(shapes_face[0].part(16 - (i + 1)).x(), shapes_face[0].part(16 - (i + 1)).y()), green, 5, 8, 0);
	}


	//Update face // Wrong somethings, but most of the time is worhty
	CorrectFaceDetection(shapes_face);

	detectNecessaryPointsOfFace(shapes_face);
	circle(face_detection_frame, left_cheek, 5, green, -1, 8);
	circle(face_detection_frame, right_cheek, 5, green, -1, 8);
	circle(face_detection_frame, top_nose, 5, green, -1, 8);
	circle(face_detection_frame, nose, 5, green, -1, 8);
	circle(face_detection_frame, chin, 5, green, -1, 8);
	circle(face_detection_frame, symmetric_point, 5, green, -1, 8);
	circle(face_detection_frame, upper_symmetric_point, 5, green, -1, 8);


	double left_eye_width = shapes_face[0].part(39).x() - shapes_face[0].part(36).x();
	if (checking_block == 0 || abs(left_eye_width * 1 / 3 - checking_block) >= 50)
	{
		checking_block = left_eye_width * 1.3 / 2;
	}

	

	//-----------------------------	Preprocess a part of image to speed up ---------------------------
	clock_t tmp01 = clock();

	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x);

	Point2f pA = Point2f(max(left_cheek.x - range_of_shoulder_sample*2.5, 0.0), min(left_cheek.y, right_cheek.y));
	Point2f pB = Point2f(min(right_cheek.x + range_of_shoulder_sample*2.5, double(frame.cols)), pA.y);
	Point2f pC = Point2f(pB.x, min(symmetric_point.y, float(frame.rows)));
	Point2f pD = Point2f(pA.x, pC.y);

	//delete unneccessary part ==>improve later
	//Mat deleted_frame = RemoveUnneccessaryImage(src);

	Mat sub_frame = src(cv::Rect(pA.x, pA.y, pB.x - pA.x, pD.y - pA.y));

	Mat CannyWithoutBlurAndMorphology = Preprocessing(sub_frame);

	//Add preprocessed part to a frame that is in the same size with the old one
	Mat BiggerCannyWithoutBlurAndMorphology(frame.rows, frame.cols, CV_8UC1, Scalar(0));
	CannyWithoutBlurAndMorphology.copyTo(BiggerCannyWithoutBlurAndMorphology(cv::Rect(pA.x, pA.y, CannyWithoutBlurAndMorphology.cols, CannyWithoutBlurAndMorphology.rows)));

	clock_t tmp02 = clock();
	std::cout << " Time for Preprocess: " << float(tmp02 - tmp01) / CLOCKS_PER_SEC << endl;

	//-----------------------------skin---------------------------
	Mat LightUpCrop = ChangeBrightness(src, 50);
	Mat skin = GetSkin(LightUpCrop);
	Mat mask_skin;
	cvtColor(skin, mask_skin, CV_BGR2GRAY);
	threshold(mask_skin, mask_skin, 0, 255, THRESH_BINARY);

	//-----------------------------neck---------------------------
	int angle_neck_left = -100;
	int angle_neck_right = -80;
	cv::vector<Point2f> leftNeckLine = DetectNeckLines(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, mask_skin, shapes_face, true, angle_neck_left);
	cv::vector<Point2f> rightNeckLine = DetectNeckLines(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, mask_skin, shapes_face, false, angle_neck_right);


	//-----------------------------collect Color Shoulder---------------------------
	collectColorShoulder(face_detection_frame);

	//-----------------------------shoulders---------------------------
	int angle_left = -150;
	int angle_right = -30;
	Mat face_detection_frame_Blur_NoCheck = face_detection_frame.clone();

	cv::vector<Point2f> leftShouderLine = detectShoulderLine(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, 
																true, angle_left, green, true, false); // isTesting
	cv::vector<Point2f> rightShouderLine = detectShoulderLine(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, 
		false, angle_right, green, true, false); //isTesting


	current_shoulderLine = vector<vector<Point2f>> {leftShouderLine, rightShouderLine};
	simplifized_current_shoulderLine = SimplifizeResult(current_shoulderLine);

	//Compare result to ground truth
	vector<double> percentages = CompareToGroundTruth(current_shoulderLine);

	//-----------------------------testing shoulder---------------------------
	if (isTesting) {
		detectShoulderLine(face_detection_frame_Blur_NoCheck, detected_edges, true, angle_left, blue, false, false);
		detectShoulderLine(face_detection_frame_Blur_NoCheck, detected_edges, false, angle_right, blue, false, false);

		cv::namedWindow("Blur_NoCheck", CV_WINDOW_NORMAL);
		cv::resizeWindow("Blur_NoCheck", 530, 700);

		cv::imshow("Blur_NoCheck", face_detection_frame_Blur_NoCheck);

		cv::namedWindow("Source_NoBlur_Check", CV_WINDOW_NORMAL);
		cv::resizeWindow("Source_NoBlur_Check", 530, 700);
		cv::imshow("Source_NoBlur_Check", face_detection_frame);
		cv::imshow("Canny Only", BiggerCannyWithoutBlurAndMorphology);

		//Combine2MatSideBySide
		cvtColor(detected_edges, detected_edges, CV_GRAY2RGB);
		Mat combine = Combine2MatSideBySide(face_detection_frame_Blur_NoCheck, detected_edges);
		returnMats.push_back(combine);
	}

	//Return face_detection_frame
	if (DebugLine) {
		frame = face_detection_frame.clone();
	}
	else {
		frame = src.clone();
	}

	std::cout << " Time for Postprocess: " << float(clock() - tmp02) / CLOCKS_PER_SEC << endl;
	return returnMats;
}



void MYcppGui::AddSticker(Mat &frame) {
	Point stickerPosition;
	double distanceMoving = checking_block / 10;
	bool need_to_crop_sticker = false;
	Mat sticker = stickerFrames[index_stickerFrames];
	//at first we take checking_block*4 as width
	double stickerWidth = checking_block * 4;
	double stickerHeight = stickerWidth / sticker.size().width * sticker.size().height;

	double new_block_checking = EuclideanDistance(nose, top_nose);
	double left_neck = nose.x - 2.5*checking_block;
	double right_neck = nose.x + 2.5 * checking_block;

	
		//First set up
		if (nth == 1) {
			int side;

			if (stickerDirection == RIGHT) {
				side = LEFT_LINE;
			}
			if (stickerDirection == LEFT) {
				side = RIGHT_LINE;
			}

			if (current_shoulderLine[side].size() != 0) {
				//At first we take a middle point
				int index_sticker = current_shoulderLine[side].size() - 1;
				stickerPosition = current_shoulderLine[side][index_sticker];

				//Get relativePostion_sticker
				relativePostion_sticker = stickerPosition.x - nose.x;
			}
		}

 		double actualPostion_sticker = relativePostion_sticker + nose.x;
		double central_point = actualPostion_sticker + stickerWidth/2;

		//The case that sticker should be in the middle of neck ==> need to be croped
		// Disappearing
		if (left_neck - stickerWidth < actualPostion_sticker && actualPostion_sticker < right_neck) {
			if (stickerStatus == Walking && Disappeared == false) {
				//in_cropping_process = true;
				stickerStatus = Disappearing;
				//relativePostion_sticker = actualPostion_sticker - nose.x;
			}
		}

		// Appearing
		if (x_ROI_sticker_begin >= (stickerWidth - distanceMoving) && stickerStatus == Disappearing) {
			//in_cropping_process = true;
			stickerStatus = Appearing;
			x_ROI_sticker_begin = 0;	//use as stickerWidth when Appearing
			//relativePostion_sticker = left_neck - nose.x - distanceMoving;		//Initial the first start when Appearing: distanceMoving
			//actualPostion_sticker = relativePostion_sticker + nose.x;		//refactor later
			Disappeared = true;
		}

		//Walking
		if (x_ROI_sticker_begin >= (stickerWidth - distanceMoving) && stickerStatus == Appearing) {
			stickerStatus = Walking;
			if (stickerDirection == LEFT) {
				actualPostion_sticker = left_neck - stickerWidth;
			}

			if (stickerDirection == RIGHT) {
				actualPostion_sticker = right_neck;
			}
			
			central_point = actualPostion_sticker + stickerWidth / 2;
			relativePostion_sticker = actualPostion_sticker - nose.x;
			x_ROI_sticker_begin = 0;
		}

		if (stickerStatus == Disappearing) {
			//Update x_ROI_sticker_begin
			x_ROI_sticker_begin += distanceMoving;
			need_to_crop_sticker = true;
			if (stickerDirection == LEFT) {
				stickerPosition = Point2f(right_neck, current_shoulderLine[RIGHT_LINE].back().y);
			}
			if (stickerDirection == RIGHT) {
				stickerPosition = Point2f(left_neck - (stickerWidth - x_ROI_sticker_begin), current_shoulderLine[LEFT_LINE].back().y);
			}
		}

		if (stickerStatus == Appearing) {
			//Update x_ROI_sticker_begin
			x_ROI_sticker_begin += distanceMoving; 
			need_to_crop_sticker = true;
			if (stickerDirection == LEFT) {
				stickerPosition = Point2f(left_neck - x_ROI_sticker_begin, current_shoulderLine[LEFT_LINE].back().y);
			}
			if (stickerDirection == RIGHT) {
				stickerPosition = Point2f(right_neck, current_shoulderLine[RIGHT_LINE].back().y);
			}
			//relativePostion_sticker -= distanceMoving;	//Move
		}


		if (stickerStatus == Walking) {
			//Find y value of postion sticker
			int side;
			if (relativePostion_sticker > 0) {
				side = RIGHT_LINE;
			}
			else {
				side = LEFT_LINE;
			}

			//The case that sticker is between 2 shoulder lines
			//With Walking, at first, we use central_point

			if (current_shoulderLine[LEFT_LINE].back().x < central_point && central_point < current_shoulderLine[RIGHT_LINE].back().x) {
				stickerPosition = Point2f(central_point - stickerWidth/2, current_shoulderLine[side].back().y);
				relativePostion_sticker += distanceMoving * stickerDirection;
			}
			//The case that the sticker go out of shoulder 
			else if (current_shoulderLine[LEFT_LINE][0].x > central_point || central_point > current_shoulderLine[RIGHT_LINE][0].x) {
				//The sticker will stay at the last point of shoulder
				stickerPosition = Point2f(current_shoulderLine[side][0].x - stickerWidth / 2, current_shoulderLine[side][0].y);
			}
			else {
				for (int i = 0; i < current_shoulderLine[side].size() - 1; i++) {

					if ((central_point - current_shoulderLine[side][i].x)*(central_point - current_shoulderLine[side][i + 1].x) <= 0) {
						float y = FindY_LineEquationThroughTwoPoint(central_point,
																	current_shoulderLine[side][i], current_shoulderLine[side][i + 1]);
						stickerPosition = Point2f(central_point - stickerWidth / 2, y);

						//Update relativePostion_sticker
						relativePostion_sticker += distanceMoving * stickerDirection;
						break;
					}
				}
			}
		}
		
		//The cases that sticker go out of the iamge
		//if (stickerPosition.x + stickerWidth / 2 > frame.size().width) {
		//	stickerWidth = frame.size().width - stickerPosition.x + stickerWidth / 2;
		//	need_to_crop_sticker = true;
		//}
		//if (stickerPosition.y - stickerHeight / 2 > frame.size().height) {
		//	stickerHeight = frame.size().height - stickerPosition.x + stickerHeight / 2;
		//	need_to_crop_sticker = true;
		//}

		//Crop the sticker
		if (need_to_crop_sticker) {
			// Setup a rectangle to define your region of interest
			double ratio = sticker.size().width / stickerWidth;		// > 1

			if (stickerStatus == Disappearing) {
				stickerWidth -= x_ROI_sticker_begin;	//Used for croping BG later
				cv::Rect stickerROI(x_ROI_sticker_begin * ratio, 0, sticker.size().width - x_ROI_sticker_begin * ratio, sticker.size().height);

				// Crop the full image to that image contained by the rectangle myROI
				// Note that this doesn't copy the data
				//sticker = sticker(stickerROI);

				cv::Mat croppedSticker = sticker(stickerROI);
				sticker = croppedSticker.clone();
			}

			if (stickerStatus == Appearing) {
				stickerWidth = x_ROI_sticker_begin;
				cv::Rect stickerROI(0, 0, stickerWidth * ratio, sticker.size().height);

				//Refactor later
				cv::Mat croppedSticker = sticker(stickerROI);
				sticker = croppedSticker.clone();
			}
		}

		//flip
		if (stickerDirection == RIGHT) {
			cv::Mat dst;               // dst must be a different Mat
			cv::flip(sticker, dst, 1);     // because you can't flip in-place (leads to segfault)
			sticker = dst.clone();
		}

		// Turn back when it hits the border || out of shoulder lines
		if (((stickerPosition.x <= 0 || stickerPosition.x + stickerWidth / 2 <= current_shoulderLine[LEFT_LINE][0].x) && stickerDirection == LEFT)
			||
			((stickerPosition.x + stickerWidth >= frame.size().width || stickerPosition.x + stickerWidth / 2 >= current_shoulderLine[RIGHT_LINE][0].x) && stickerDirection == RIGHT))
		{
			stickerDirection = -1 * stickerDirection; // change direction
			actualPostion_sticker = stickerPosition.x + stickerDirection * 5;		// 1 is in case
			relativePostion_sticker = actualPostion_sticker - nose.x;
			Disappeared = false;
			return;
		}

		Mat cropedBG;
		
		Rect roi(stickerPosition.x, stickerPosition.y - stickerHeight / 2, stickerWidth, stickerHeight);	
		frame(roi).copyTo(cropedBG);
		Mat cloneCropedBG = cropedBG.clone();
		Mat dst = overlrayImage(cropedBG, sticker);

		dst.copyTo(frame(roi));

		//Try to add Mask overlay
		//if (stickerStatus == Appearing || stickerStatus == Disappearing) {
		//	/*Mat cropedBG02;
		//	Rect roi(left_cheek.x, left_cheek.y, right_cheek.x - left_cheek.x, chin.y + 2*checking_block - left_cheek.y);
		//	frame(roi).copyTo(cropedBG02);*/
		//	Mat LightUpCrop = ChangeBrightness(cloneCropedBG);
		//	Mat skin = GetSkin(LightUpCrop);

		//	Mat mask1, mask2;
		//	mask1 = skin.clone();
		//	cvtColor(mask1, mask2, CV_BGR2GRAY);
		//	threshold(mask2, mask2, 0, 255, THRESH_BINARY);

		//	overlayMask(dst, cloneCropedBG, mask2);
		//	dst.copyTo(frame(roi));
		//}
}

Mat MYcppGui::ImageProcessing(string fileName, vector<cv::Point2f> userInput)
{
	cv::Mat img_input = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
	ImageProcessing_Final(img_input, false, true, true);
	return img_input;
}

void MYcppGui::Morphology_Operations(Mat &src)
{
	int morph_elem = 0;

	//kernel for morphology blur
	int morph_size = 10;

	// Since MORPH_X : 2,3,4,5 and 6
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

	/// Apply the specified morphology operation
	morphologyEx(src, src, MORPH_OPEN, element);	//open
	morphologyEx(src, src, MORPH_CLOSE, element);	//close
}

void MYcppGui::CannyProcessing(Mat image, OutputArray edges)
{
	//Min threshold for Canny 16
	int CannyThreshold = 20;
	
	//unknow Canny stat
	int ratio = 3;
	
	//unknow Canny stat
	int kernel_size = 3;

	Mat dst, frame_gray;
	/// Create a matrix of the same type and size as image (for dst)
	dst.create(image.size(), image.type());

	/// Convert the image to grayscale
	cvtColor(image, frame_gray, CV_BGR2GRAY);

	/// Reduce noise with a kernel 3x3
	blur(frame_gray, edges, Size(3, 3));
	
	/// Canny detector
	Canny(edges, edges, CannyThreshold, CannyThreshold*ratio, kernel_size);


	/// Using Canny's output as a mask, we display our result
	//dst = Scalar::all(0);
	//image.copyTo(dst, edges); // check soon
}

std::vector<dlib::full_object_detection> MYcppGui::face_detection_update(Mat frame)
{
	Mat src;
	src = frame;

	// Resize image for face detection
	cv::Mat frame_small;
	cv::resize(frame, frame_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);
	
	// Change to dlib's image format. No memory is copied.
	dlib::cv_image<dlib::bgr_pixel> cimg_small(frame_small);
	dlib::cv_image<dlib::bgr_pixel> cimg(src);

	// We need a face detector.  We will use this to get bounding boxes for
	// each face in an image.
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	// Now tell the face detector to give us a list of bounding boxes
	// around all the faces in the image.
	clock_t tmp = clock();
	
	// Detect faces on resize image
	if (nth % SKIP_FRAMES == 1)
	{
		cur_dets = detector(cimg_small);
	}

	std::cout << " Time for detect face: " << float(clock() - tmp) / CLOCKS_PER_SEC << endl;

	cout << "Number of faces detected: " << cur_dets.size() << endl;

	tmp = clock();
	// Find the pose of each face.
	std::vector<dlib::full_object_detection> new_shapes;
	for (unsigned long j = 0; j < cur_dets.size(); ++j)
	{
		// Resize obtained rectangle for full resolution image.
		dlib::rectangle r(
			(long)(cur_dets[j].left() * FACE_DOWNSAMPLE_RATIO),
			(long)(cur_dets[j].top() * FACE_DOWNSAMPLE_RATIO),
			(long)(cur_dets[j].right() * FACE_DOWNSAMPLE_RATIO),
			(long)(cur_dets[j].bottom() * FACE_DOWNSAMPLE_RATIO)
			);

		// Landmark detection on full sized image
		dlib::full_object_detection shape = shape_predictor(cimg, r);
		new_shapes.push_back(shape);
	}

	if (cur_dets.size() == 0) {
		cur_dets = detector(cimg);
		cout << "AGAIN: Number of faces detected: " << cur_dets.size() << endl;

		for (unsigned long j = 0; j < cur_dets.size(); ++j)
		{
			// Resize obtained rectangle for full resolution image.
			dlib::rectangle r(
				(long)(cur_dets[j].left()),
				(long)(cur_dets[j].top()),
				(long)(cur_dets[j].right()),
				(long)(cur_dets[j].bottom())
				);

			// Landmark detection on full sized image
			dlib::full_object_detection shape = shape_predictor(cimg, r);
			new_shapes.push_back(shape);
		}
	}
	std::cout << " Time for landmark face: " << float(clock() - tmp) / CLOCKS_PER_SEC << endl;

	//std::getchar();
	return new_shapes;
}

std::vector<dlib::full_object_detection> MYcppGui::face_detection_dlib_image(Mat frame)
{
	// We need a face detector.  We will use this to get bounding boxes for
	// each face in an image.
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	Mat src;

	//cv::resize(frame, src, cv::Size(), 0.75, 0.75);
	src = frame;

	//dlib::array2d<dlib::rgb_pixel> cimg;
	//assign_image(cimg, dlib::cv_image<dlib::bgr_pixel>(src));
	dlib::cv_image<dlib::bgr_pixel> cimg(src);
	//pyramid_up(cimg);

	// Now tell the face detector to give us a list of bounding boxes
	// around all the faces in the image.
	//cout << cimg.size() << endl;
	clock_t tmp = clock();
	std::vector<dlib::rectangle> dets = detector(cimg);
	std::cout << " Time for detect face: " << float(clock() - tmp) / CLOCKS_PER_SEC << endl;

	cout << "Number of faces detected: " << dets.size() << endl;

	// Now we will go ask the shape_predictor to tell us the pose of
	// each face we detected.
	std::vector<dlib::full_object_detection> new_shapes;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		dlib::full_object_detection shape = shape_predictor(cimg, dets[j]);
		//cout << "number of parts: " << shape.num_parts() << endl;
		//for (int i = 0; i < shape.num_parts(); i++) {
			//cout << "<part name='" << i << "' x='" << shape.part(i).x() << "' y='" << shape.part(i).y() << "'/>" << endl;
		//}
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.

		new_shapes.push_back(shape);
	}
	
	//shapes.assign(shapess.begin(), shapess.end());
	// Now let's view our face poses on the screen.
	//win.clear_overlay();
	//win.set_image(cimg);
	//win.add_overlay(render_face_detections(*shapes));

	// We can also extract copies of each face that are cropped, rotated upright,
	// and scaled to a standard size as shown here:
	//dlib::array<array2d<rgb_pixel> > face_chips;
	//extract_image_chips(cimg, get_face_chip_details(*shapes), face_chips);
	//win_faces.set_image(tile_images(face_chips));


	//std::getchar();
	return new_shapes;
}
void MYcppGui::CorrectFaceDetection(std::vector<dlib::full_object_detection>& shapes_face) {
	Point2f left_point = Point2f(shapes_face[0].part(3).x(), shapes_face[0].part(3).y());
	Point2f right_point = Point2f(shapes_face[0].part(13).x(), shapes_face[0].part(13).y());
	Point2f centre_point = Point2f(shapes_face[0].part(33).x(), shapes_face[0].part(33).y());
	Point2f chin = Point2f(shapes_face[0].part(8).x(), shapes_face[0].part(8).y());
	Point2f top_nose = Point2f(shapes_face[0].part(27).x(), shapes_face[0].part(27).y());

	double distance_3_33 = EuclideanDistance(left_point, centre_point);
	double distance_13_33 = EuclideanDistance(right_point, centre_point);
	bool fix_left;


	//left wrong
	if (distance_3_33 - distance_13_33 > checking_block / 3) {		//at first checking_block/3
		fix_left = true;
	}
	else if (distance_13_33 - distance_3_33 > checking_block / 3) {
		fix_left = false;
	}
	else {
		return;
	}

	for (int i = 0; i < 8; i++) {
		left_point = Point2f(shapes_face[0].part(i).x(), shapes_face[0].part(i).y());
		right_point = Point2f(shapes_face[0].part(16 - i).x(), shapes_face[0].part(16 - i).y());

		Point2f fixedPoint;
		if (fix_left) {		
			fixedPoint = mirror(right_point, top_nose, chin);	//top_nose to chin is a mirror
			shapes_face[0].part(i) = dlib::point(fixedPoint.x, fixedPoint.y);
		}
		else {
			fixedPoint = mirror(left_point, top_nose, chin);
			shapes_face[0].part(16 - i) = dlib::point(fixedPoint.x, fixedPoint.y);
		}
	}
}

//!Detect neccessary points of face for next processing
void MYcppGui::detectNecessaryPointsOfFace(std::vector<dlib::full_object_detection> shapes_face) {

	left_cheek = Point2f(shapes_face[0].part(3).x(), shapes_face[0].part(3).y());		//use to be 4
	right_cheek = Point2f(shapes_face[0].part(13).x(), shapes_face[0].part(13).y());	//used to be 12

	chin = Point2f(shapes_face[0].part(8).x(), shapes_face[0].part(8).y());
	top_nose = Point2f(shapes_face[0].part(27).x(), shapes_face[0].part(27).y());
	nose = Point2f(shapes_face[0].part(30).x(), shapes_face[0].part(30).y());
	symmetric_point = Point2f(chin.x * 2 - top_nose.x, chin.y * 2 - top_nose.y);
	upper_symmetric_point = Point2f(top_nose.x * 7 / 3 - chin.x * 4 / 3, top_nose.y * 7 / 3 - chin.y * 4 / 3);
}


Mat MYcppGui::RemoveUnneccessaryImage(Mat& frame) {
	double angle_left = -150;
	double angle_right = -30;

	double radian_left = angle_left * CV_PI / 180.0;
	double radian_right = angle_right * CV_PI / 180.0;
	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x); //used to be *2
	double length = abs(range_of_shoulder_sample / cos(radian_left));

	//move these point a bit to not effect edge detection (10px)
	Point2f head_bottom_shoulder = Point2f(chin.x, chin.y + 10);
	Point2f end_bottom_shoulder_left = Point2f(head_bottom_shoulder.x + length*cos(radian_left) + 10, head_bottom_shoulder.y - length*sin(radian_left));
	Point2f end_second_bottom_shoulder_left = Point2f(end_bottom_shoulder_left.x, symmetric_point.y);
	Point2f end_bottom_shoulder_right = Point2f(head_bottom_shoulder.x + length*cos(radian_right) - 10, head_bottom_shoulder.y - length*sin(radian_right));
	Point2f end_second_bottom_shoulder_right = Point2f(end_bottom_shoulder_right.x, symmetric_point.y);
	vector<Point> vertices01{ end_second_bottom_shoulder_left, end_bottom_shoulder_left, head_bottom_shoulder, end_bottom_shoulder_right, end_second_bottom_shoulder_right };

	Point2f head_cheeck_remover_left = Point2f(left_cheek.x - 10, left_cheek.y);
	Point2f end_cheeck_remover_left_01 = Point2f(0, left_cheek.y);
	Point2f end_cheeck_remover_left_02 = Point2f(0, left_cheek.y + left_cheek.x / sqrt(3));
	vector<Point> vertices02{ head_cheeck_remover_left, end_cheeck_remover_left_01, end_cheeck_remover_left_02 };

	Point2f head_cheeck_remover_right = Point2f(right_cheek.x + 10, right_cheek.y);
	Point2f end_cheeck_remover_right_01 = Point2f(frame.size().width, right_cheek.y);
	Point2f end_cheeck_remover_right_02 = Point2f(frame.size().width, right_cheek.y + (frame.size().width - right_cheek.x) / sqrt(3));
	vector<Point> vertices03{ head_cheeck_remover_right, end_cheeck_remover_right_01, end_cheeck_remover_right_02 };

	Mat dst = frame.clone();
	vector<vector<Point>> pts{ vertices01, vertices02, vertices03 };

	fillPoly(dst, pts, black);
	return dst;
}

Vec3b AverageColor(Mat& frame, double x, double y) {
	Vec3b color;

	//Border cases
	if (x - 3 < 0 || y - 3 < 0 || x + 3 > frame.size().width || y + 3 > frame.size().height) {		// width --, ex 720 ==> only 719
		if (x < 0) {
			x = 0;
		}
		if (y < 0) {
			y = 0;
		}
		if (x > frame.size().width - 1){
			x  = frame.size().width - 1;
		}
		if (y > frame.size().height - 1) {
			y = frame.size().height - 1;
		}
		color = frame.at<Vec3b>(Point2f(x, y));
		return color;
	}

	cv::Mat roiMat = frame(cv::Rect(x - 2, y - 2, 5, 5));
	cv::Scalar mean;
	mean = cv::mean(roiMat);
	color = Vec3b(mean[0], mean[1], mean[2]);
	return color;
}

void MYcppGui::collectColor(Mat&frame, vector<Vec3b> &colorCollection, Point2f head_point, Point2f end_point, double epsilon) {
	//LineIterator 
	LineIterator it(frame, head_point, end_point, 8, false);

	for (int i = 0; i < it.count; i += 1, ++it)
	{
		if (i % 10 != 0) {	//move 10px
			continue;
		}
		int x = it.pos().x;
		int y = it.pos().y;

		//Vec3b color = frame.at<Vec3b>(Point2f(x, y));
		Vec3b color = AverageColor(frame, x, y);

		if (i == 0)
		{
			colorCollection.push_back(color);
			continue;
		}

		double minColourDistance = ColourDistance(color, colorCollection[0]);
		//cout << minColourDistance << endl;

		for (int j = 1; j < colorCollection.size(); j++)
		{
			double colourDistance = ColourDistance(color, colorCollection[j]);
			//cout << colourDistance << endl;
			if (colourDistance < minColourDistance)
				minColourDistance = colourDistance;
		}

		circle(frame, Point2f(x, y), 2, green, -1, 8);

		if (minColourDistance > epsilon)
		{
			colorCollection.push_back(color);
			circle(frame, Point2f(x, y), 5, blue, -1, 8);
		}
		//cout << "-----------------" << endl;
	}

}

void MYcppGui::collectColorShoulder(Mat& frame) {
	double angle_left = -150;
	double angle_right = -30;

	double radian_left = angle_left * CV_PI / 180.0;
	double radian_right = angle_right * CV_PI / 180.0;
	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x); //used to be *2
	double length = abs(range_of_shoulder_sample / cos(radian_left));

	//move these point a bit to not effect edge detection (10px)
	
	for (int i = -1; i <= 1; i += 2) {
		Point2f head_bottom_shoulder = Point2f(chin.x + i*checking_block, chin.y);
		Point2f end_bottom_shoulder_left = Point2f(head_bottom_shoulder.x + length*cos(radian_left) + 10, head_bottom_shoulder.y - length*sin(radian_left));
		Point2f end_second_bottom_shoulder_left = Point2f(end_bottom_shoulder_left.x, symmetric_point.y);
		Point2f end_bottom_shoulder_right = Point2f(head_bottom_shoulder.x + length*cos(radian_right) - 10, head_bottom_shoulder.y - length*sin(radian_right));
		Point2f end_second_bottom_shoulder_right = Point2f(end_bottom_shoulder_right.x, symmetric_point.y);

		// Collect color of shoulder 
		collectColor(frame, colorCollection_Shoulder, head_bottom_shoulder, end_bottom_shoulder_left, 30);
		collectColor(frame, colorCollection_Shoulder, end_bottom_shoulder_left, end_second_bottom_shoulder_left, 30);

		collectColor(frame, colorCollection_Shoulder, head_bottom_shoulder, end_bottom_shoulder_right, 30);
		collectColor(frame, colorCollection_Shoulder, end_bottom_shoulder_right, end_second_bottom_shoulder_right, 30);
	}
	
}

Vector<Vec3b> MYcppGui::collectColorNeck(Mat&frame, Point2f head_neck, Point2f end_neck) {
	Vector<Vec3b> colorNeckCollection;
	//LineIterator 
	LineIterator it(frame, head_neck, end_neck, 8, false);
	
	for (int i = 0; i < it.count; i += 1, ++it)
	{
		if (i % 10 != 0) {	//move 10px
			continue;
		}
		int x = it.pos().x;
		int y = it.pos().y;

		//Vec3b color = frame.at<Vec3b>(Point2f(x, y));
		Vec3b color = AverageColor(frame, x, y);

		if (i == 0)
		{
			colorNeckCollection.push_back(color);
			continue;
		}

		double minColourDistance = ColourDistance(color, colorNeckCollection[0]);
		cout << minColourDistance << endl;

		for (int j = 1; j < colorNeckCollection.size(); j++)
		{
			double colourDistance = ColourDistance(color, colorNeckCollection[j]);
			cout << colourDistance << endl;
			if (colourDistance < minColourDistance)
				minColourDistance = colourDistance;
		}

		circle(frame, Point2f(x, y), 5, green, -1, 8);

		if (minColourDistance > 20)
		{
			colorNeckCollection.push_back(color);
			circle(frame, Point2f(x, y), 5, blue, -1, 8);
		}

		cout << "-----------------" << endl;
	}
	return colorNeckCollection;
}



cv::vector<Point2f> MYcppGui::DetectNeckLines(Mat shoulder_detection_image, Mat detected_edges, Mat mask_skin, std::vector<dlib::full_object_detection> shapes_face, bool leftHandSide, int angle_neck) {
	
	Point2f head_neck;
	Point2f end_neck;
	Point2f head_neck02;
	Point2f end_neck02;
	Vector<Vec3b> colors;

	if (leftHandSide){
		head_neck = Point2f(left_cheek.x, shapes_face[0].part(4).y());
		head_neck02 = Point2f(shapes_face[0].part(6).x() + 5, head_neck.y);		//10 is just in case
		end_neck = Point2f(head_neck.x, chin.y + checking_block);
		end_neck02 = Point2f(head_neck02.x, end_neck.y);

		//collect color of skin this area		//Use later
		//colors = collectColorNeck(shoulder_detection_image, head_neck02, end_neck02);

		//Point2f point04 = Point2f(shapes_face[0].part(4).x() + 10, shapes_face[0].part(4).y()); //5 is just in case
		//Point2f point06 = Point2f(shapes_face[0].part(6).x() + 5, shapes_face[0].part(6).y());
		//Vector<Vec3b> Morecolors = collectColorNeck(shoulder_detection_image, point04, point06);
		//for (int i = 0; i < Morecolors.size() - 1; i++) {
		//	colors.push_back(Morecolors[i]);
		//}
	}
	else {
		head_neck = Point2f(shapes_face[0].part(10).x() - 5, shapes_face[0].part(12).y());
		head_neck02 = Point2f(right_cheek.x, head_neck.y);		//10 is just in case
		end_neck = Point2f(head_neck.x, chin.y + checking_block);
		end_neck02 = Point2f(head_neck02.x, end_neck.y);

		//collect color of skin this area
		//colors = collectColorNeck(shoulder_detection_image, head_neck, end_neck);
		//Point2f point12  = Point2f(shapes_face[0].part(12).x() - 10, shapes_face[0].part(12).y()); //5 is just in case
		//Point2f point10  = Point2f(shapes_face[0].part(10).x() - 5, shapes_face[0].part(10).y());
		//Vector<Vec3b> Morecolors = collectColorNeck(shoulder_detection_image, point12, point10);
		//for (int i = 0; i < Morecolors.size() - 1; i++) {
		//	colors.push_back(Morecolors[i]);
		//}
	}

	
	line(shoulder_detection_image, head_neck, end_neck, red, 3, 8, 0);
	line(shoulder_detection_image, head_neck02, end_neck02, red, 3, 8, 0);
	
	int value_in_edge_map = 0;
	cv::vector<cv::vector<Point2f>> point_collection;

	for (int i = 0; head_neck.y + i*checking_block / 2 < end_neck.y; i++) {
		double Y = head_neck.y + i*checking_block / 2;
		cv::vector<Point2f> point_line;

		double distance_01_to_02 = abs(head_neck.x - head_neck02.x);
		for (int j = 0; j * 2 < distance_01_to_02; j++) {
			double X = head_neck.x + j * 2;
			value_in_edge_map = detected_edges.at<uchar>(Y, X);	// y first, x later
			
			if (value_in_edge_map == 255) {
				//Color check 
				//Vec3b color = shoulder_detection_image.at<Vec3b>(Point2f(X, Y));
				//Vec3b color = AverageColor(shoulder_detection_image, X, Y);
				//bool is_match_color = IsMatchColor(color, colors, 30);		//30 is good but sometimes too much

				int color_skin = mask_skin.at<uchar>(Point2f(X, Y));
				//int color_skin02 = mask_skin.at<uchar>(Point2f(Y, X));

				if (color_skin == 255) {
					if (point_line.empty() || EuclideanDistance(Point2f(X, Y), point_line.back()) >= 3)	//10 
					{
						point_line.push_back(Point2f(X, Y));
					}
				}				
			}
		}
		point_collection.push_back(point_line);
	}

	//Just for testing and debug
	for (int a = 0; a < point_collection.size() - 1; a++) {
		for (int b1 = 0; b1 < point_collection[a].size(); b1++)
		{
			for (int b2 = 0; b2 < point_collection[a + 1].size(); b2++)
			{
				//Check difference of angle
				if (abs(Angle(point_collection[a][b1], point_collection[a + 1][b2]) - angle_neck) <= 15)
				{
					line(shoulder_detection_image, point_collection[a][b1], point_collection[a + 1][b2], yellow, 3, 8, 0);
				}
			}
		}
	}
	
	cv::vector<cv::vector<Point2f>> possible_lines;
	cv::vector<Point2f> neck_line;
	for (int i = 0; i < point_collection.size(); i++) {
		for (int j = 0; j < point_collection[i].size(); j++) {
			cv::vector<Point2f> path = findPath(j, i, point_collection, angle_neck);

			if (possible_lines.empty() || path.size() > possible_lines.back().size())
			{
				if (!path.empty())
				{
					possible_lines.push_back(path);
				}
			}
		}
	}

	if (!possible_lines.empty()) {
		neck_line = possible_lines.back();

		for (int i = 0; i < neck_line.size() - 1; i++)
		{
			line(shoulder_detection_image, neck_line[i], neck_line[i + 1], green, 5, 8, 0);
		}	
	}
	return neck_line;
}

//new fuction
cv::vector<Point2f> MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, bool leftHandSide, int angle, Scalar color, bool checkColor, bool checkPreviousResult)
{
	/*Mat LAB_frame;
	cvtColor(shoulder_detection_image, LAB_frame, CV_BGR2Lab);*/

	double radian = angle * CV_PI / 180.0;
	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x); //used to be *2
	double length = abs(range_of_shoulder_sample / cos(radian));
	
	//bottom shoulder sample line
	Point2f head_bottom_shoulder = chin;
	Point2f end_bottom_shoulder = Point2f(head_bottom_shoulder.x + length*cos(radian), head_bottom_shoulder.y - length*sin(radian));
	Point2f end_second_bottom_shoulder = Point2f(end_bottom_shoulder.x, symmetric_point.y);
	
	//upper shoulder sample line
	Point2f head_upper_shoulder;

	int sign = 1;		//GO RIGHT
	if (leftHandSide) {	//GO LEFT
		sign = -1;
	}

	if (leftHandSide) {	//GO LEFT
		head_upper_shoulder = Point2f(left_cheek.x - 10, left_cheek.y + 10);	// - distance_from_face_to_shouldersample
	}
	else {
		head_upper_shoulder = Point2f(right_cheek.x + 10, right_cheek.y + 10);	// + distance_from_face_to_shouldersample
	}
	

	Point2f end_upper_shoulder = Point2f(head_upper_shoulder.x + length*2.5*cos(radian), head_upper_shoulder.y - length*2.5*sin(radian));

	// Draw DebugLines
	line(shoulder_detection_image, head_bottom_shoulder, end_bottom_shoulder, red, 3, 8, 0);
	line(shoulder_detection_image, end_bottom_shoulder, end_second_bottom_shoulder, red, 3, 8, 0);
	line(shoulder_detection_image, head_upper_shoulder, end_upper_shoulder, red, 3, 8, 0);

	cv::vector<cv::vector<Point2f>> point_collection;
	
	//intersection_point_01 to mark where the second bottom shoulder line start
	Point2f intersection_point_01;
	intersection(head_upper_shoulder, end_upper_shoulder, symmetric_point, end_bottom_shoulder, intersection_point_01);

	
	//Count points added
	int COUNT_ALL = 0;

	Point2f intersection_point_with_previous_result;

	//Take points on shoulder_sample follow "checking_block" and build LineIterator from these point to symmetric_point (but stop at bottom shoulder_line
	for (int j = 0; abs(checking_block*j*cos(radian)) < range_of_shoulder_sample*2.5; j++) {
		Point2f current_point = Point2f(head_upper_shoulder.x + checking_block*j*cos(radian), head_upper_shoulder.y - checking_block*j*sin(radian));
		int value_in_edge_map = 0;
		Point2f intersection_point;

		//Find intersection_point which is point to stop the LineIterator ==> Create LineIterator fo
		if (current_point.y <= intersection_point_01.y) {
			intersection(head_bottom_shoulder, end_bottom_shoulder, current_point, symmetric_point, intersection_point);
		}
		else {
			intersection(end_bottom_shoulder, end_second_bottom_shoulder, current_point, symmetric_point, intersection_point);
		}

		//Got error when run train_23. So i let it stop when touch end_second_bottom_shoulder
		if (intersection_point.y > end_second_bottom_shoulder.y){
			break;
		}

		if (checkPreviousResult) {
			//Find intersection between "it" and Simplifized User Input

			//THe case we re between 2 shoulder lines
			if (j == 0) {
				bool check_intersected = doIntersect(intersection_point, current_point, simplifized_current_shoulderLine[!leftHandSide].back(),
					simplifized_current_shoulderLine[!leftHandSide][simplifized_current_shoulderLine[!leftHandSide].size() - 2]);
				if (!check_intersected) {
					Point2f intersection_point02;
					intersection(intersection_point, current_point, simplifized_current_shoulderLine[!leftHandSide].back(),
						simplifized_current_shoulderLine[!leftHandSide][simplifized_current_shoulderLine[!leftHandSide].size() - 2],
						intersection_point02);
					simplifized_current_shoulderLine[!leftHandSide].push_back(intersection_point02);
				}
			}

			//If the no suitable result, intersection_point_with_previous_result will be the nearest point to neck
			intersection_point_with_previous_result = simplifized_current_shoulderLine[!leftHandSide].back();

			bool is_intersect = false;
			//left hand simplifized_current_shoulderLine is 0 which == !leftHandSide and otherwises
			for (int i = 0; i < simplifized_current_shoulderLine[!leftHandSide].size() - 1; i++) {
				if (doIntersect(intersection_point, current_point, simplifized_current_shoulderLine[!leftHandSide][i], simplifized_current_shoulderLine[!leftHandSide][i + 1]))
				{
					intersection(intersection_point, current_point, simplifized_current_shoulderLine[!leftHandSide][i], simplifized_current_shoulderLine[!leftHandSide][i + 1]
						, intersection_point_with_previous_result);
					is_intersect = true;
					break;
				}
			}

			//THe case we re out of the shoulder line
			if (!is_intersect) {
				intersection(intersection_point, current_point, simplifized_current_shoulderLine[!leftHandSide][0], simplifized_current_shoulderLine[!leftHandSide][1]
					, intersection_point_with_previous_result);
			}
			//circle(shoulder_detection_image, intersection_point_with_previous_result, 10, green, -1, 8);
		}


		cv::vector<Point2f> point_line;
		//LineIterator from  intersection_point which we found above to upper shoulder line 		//Go inside out
		LineIterator it(shoulder_detection_image, intersection_point, current_point, 8, false);

		//Get all intersections of LineIterator and Canny lines;
		for (int i = 0; i < it.count; i+=2, ++it, ++it)
		{
			// Skip the point out of the frame
			if (it.pos().x < 0 || it.pos().x > shoulder_detection_image.size().width - 1|| it.pos().y > shoulder_detection_image.size().height - 1) {
				continue;
			}

			value_in_edge_map = detected_edges.at<uchar>(it.pos().y, it.pos().x);	// y first, x later
			Point2f current_point = Point2f(it.pos().x, it.pos().y);

			//problem only on train_10
			if (it.pos().y + 15 > shoulder_detection_image.size().height - 1)
				break;
			
			//Color check //update later
			//Vec3b color = shoulder_detection_image.at<Vec3b>(Point2f(it.pos().x, it.pos().y + 25));
			/*Vec3b color = LAB_frame.at<Vec3b>(Point2f(it.pos().x, it.pos().y + 25));
			Vec3f color_LAB = Vec3f((float)color[0] / 255 * 100, (float)color[1] - 128, (float)color[2] - 128);*/

			// go inside body ==> - sign*20
			//Vec3b color_inside = AverageColor(shoulder_detection_image, it.pos().x - sign*20, it.pos().y + 20);		//used to be 25 -- 10 is good
			Vec3b color_inside = AverageColor(shoulder_detection_image, it.pos().x, it.pos().y + 18);
			// go outside body ==> + sign*20
			Vec3b color_outside = AverageColor(shoulder_detection_image, it.pos().x + sign * 20, it.pos().y - 20);	
			Vec3b color_outside02 = AverageColor(shoulder_detection_image, it.pos().x, it.pos().y - 20);

			bool is_unmatch_color_outside01 = !IsMatchColor(color_outside, colorCollection_Shoulder, 30);
			bool is_unmatch_color_outside02 = !IsMatchColor(color_outside, colorCollection_Shoulder, 30);
			bool is_unmatch_color_outside = is_unmatch_color_outside01 || is_unmatch_color_outside02;

			bool is_match_color = IsMatchColor(color_inside, colorCollection_Shoulder, 30);	//belong to inside
			//&& !IsMatchColor(color_outside, colorCollection_Shoulder, 30);				// not belong to outside
			if (!checkColor)
				is_match_color = true;

			//check PreviousResult
			bool is_close_to_previous_result = true;
			if (checkPreviousResult) {
				if (EuclideanDistance(current_point, intersection_point_with_previous_result) > checking_block * 1.5) {	//At first, I take checking_block*1.5
					is_close_to_previous_result = false;
				}
				else {
					circle(shoulder_detection_image, Point2f(it.pos().x, it.pos().y), 1, black, -1, 8);
				}
				
			}

			if (value_in_edge_map == 255 && is_match_color)
			{
				if (point_line.empty() || EuclideanDistance(current_point, point_line.back()) >= checking_block / 2.5)	//10 work really well - 15 works well too
				{
					circle(shoulder_detection_image, Point2f(it.pos().x, it.pos().y), 5, red, -1, 8);

					if (is_close_to_previous_result){
						//circle(shoulder_detection_image, Point2f(it.pos().x, it.pos().y), 7, blue, -1, 8);
						point_line.push_back(Point2f(it.pos().x, it.pos().y));
						COUNT_ALL++;
					}
				}
			}
		}
		point_collection.push_back(point_line);
	}

	

	//Reduce point_collection in the case that shirt texture is too complicated
	if (COUNT_ALL > 105) {		//fix 83-84
		cout << COUNT_ALL << " --> ";
		RefinePoint_collection(shoulder_detection_image, point_collection);
		//check COUNT_ALL again
		COUNT_ALL = 0;
		for (int j = 0; j < point_collection.size(); j++) {
			COUNT_ALL += point_collection[j].size();
		}

		if (COUNT_ALL > 75) {
			RefinePoint_collection(shoulder_detection_image, point_collection);
			COUNT_ALL = 0;
			for (int j = 0; j < point_collection.size(); j++) {
				COUNT_ALL += point_collection[j].size();
			}
		}
	}

	cout << COUNT_ALL << endl;
	////Draw simplifizedUserInput for testing
	//for (int i = 0; i < simplifizedUserInput[!leftHandSide].size(); i++) {
	//	circle(shoulder_detection_image, simplifizedUserInput[!leftHandSide][i], 5, yellow, -1, 8);
	//	if (i < simplifizedUserInput[!leftHandSide].size() - 1) {
	//		//line(shoulder_detection_image, simplifizedUserInput[!leftHandSide][i], simplifizedUserInput[!leftHandSide][i + 1], yellow, 3, 8, 0);
	//	}
	//}

	//Draw simplifized_current_shoulderLine for testing
	if (checkPreviousResult || simplifized_current_shoulderLine.size() > 0) {
		for (int i = 0; i < simplifized_current_shoulderLine[!leftHandSide].size(); i++) {
			circle(shoulder_detection_image, simplifized_current_shoulderLine[!leftHandSide][i], 5, yellow, -1, 8);
			if (i < simplifized_current_shoulderLine[!leftHandSide].size() - 1) {
				//line(shoulder_detection_image, simplifized_current_shoulderLine[!leftHandSide][i], simplifized_current_shoulderLine[!leftHandSide][i + 1], yellow, 3, 8, 0);
			}
		}
	}


	//take potential point for shoulder line by checking angle of these line
	int angle_for_arm = -75;
	if (leftHandSide)
		angle_for_arm = -105;

	//check ki trc khi xoa
	for (int a = 0; a < point_collection.size() - 1; a++) {
		for (int b1 = 0; b1 < point_collection[a].size(); b1++)
		{
			for (int b2 = 0; b2 < point_collection[a + 1].size(); b2++)
			{
				//Check difference of angle
				if (abs(Angle(point_collection[a][b1], point_collection[a + 1][b2]) - angle) <= 20 || abs(Angle(point_collection[a][b1], point_collection[a + 1][b2]) - angle_for_arm) <= 20)
				{
					line(shoulder_detection_image, point_collection[a][b1], point_collection[a + 1][b2], red, 3, 8, 0);
				}
			}
		}
	}
	cv::vector<cv::vector<Point2f>> possible_lines;
	cv::vector<cv::vector<Point2f>> possible_lines_for_arm;


	for (int i = 0; i < point_collection.size(); i++) {
		for (int j = 0; j < point_collection[i].size(); j++) {
			cv::vector<Point2f> path = findPath(j, i, point_collection, angle);
			cv::vector<Point2f> path_for_arm = findPath(j, i, point_collection, angle_for_arm);

			if (possible_lines.empty() || path.size() > possible_lines.back().size())
			{
				if (!path.empty())
				{
					possible_lines.push_back(path);
				}				
			}

			if (possible_lines_for_arm.empty() || path_for_arm.size() > possible_lines_for_arm.back().size())
			{
				if (!path_for_arm.empty())
				{
					possible_lines_for_arm.push_back(path_for_arm);
					//cout << "A new max line" << endl;
				}
			}
		}
	}

	//old way use the longest one
	cv::vector<Point2f> shoulder_line_for_arm_longest;
	if (!possible_lines_for_arm.empty()) {
		shoulder_line_for_arm_longest = possible_lines_for_arm.back();

		for (int i = 0; i < shoulder_line_for_arm_longest.size() - 1; i++)
		{
			line(shoulder_detection_image, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], black, 5, 8, 0);
		}
	}



	if (!possible_lines.empty()) {
		cv::vector<Point2f> shoulder_line = possible_lines.back();

		for (int i = 0; i < shoulder_line.size() - 1; i++)
		{
			line(shoulder_detection_image, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);
		}
	

		//new way which is to detect the position of arm line
		//list use pushback, so the last one is [0]
		int index_one_third = shoulder_line.size() * 1 / 3;
		int index_half = shoulder_line.size()/ 2;
		cv::vector<Point2f> shoulder_line_for_arm_test;

		for (int i = possible_lines_for_arm.size() - 1; i >= 0; i--)
		{
			if (abs(shoulder_line[0].x - possible_lines_for_arm[i].back().x) < abs(shoulder_line[0].x - shoulder_line[index_one_third].x))
			{
				if (possible_lines_for_arm[i].back().y > shoulder_line[index_half].y) {
					shoulder_line_for_arm_test = possible_lines_for_arm[i];
					break;
				}
			}
		}

		if (!shoulder_line_for_arm_test.empty())
		{
			for (int i = 0; i < shoulder_line_for_arm_test.size() - 1; i++)
			{
				line(shoulder_detection_image, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], yellow, 5, 8, 0);
			}
		}
		
		//Check if the longest arm line is overlap some points of shoulder lines
		if (CheckCommon(shoulder_line, shoulder_line_for_arm_longest)) {
			for (int i = 0; i < shoulder_line_for_arm_longest.size(); i++) {
				for (int j = 0; j < shoulder_line.size(); j++) {
					//First common points
					if (shoulder_line[j] == shoulder_line_for_arm_longest[i]) {
						if (j + 1 < shoulder_line.size() && i + 1 < shoulder_line_for_arm_longest.size()) {
							//Check if there is a second point
							if (shoulder_line[j + 1] == shoulder_line_for_arm_longest[i + 1]) {

								// Remove the first N elements, and shift everything else down by N indices
								shoulder_line.erase(shoulder_line.begin(), shoulder_line.begin() + j);
								for (int k = 2; j + k < shoulder_line.size() && i + k < shoulder_line_for_arm_longest.size(); k++) {
									if (shoulder_line[j + k] != shoulder_line_for_arm_longest[i + k]) {
										shoulder_line_for_arm_longest.erase(shoulder_line_for_arm_longest.begin() + i + k,
											shoulder_line_for_arm_longest.end());
										break;
									}
								}

								//Show up
								for (int i = 0; i < shoulder_line_for_arm_longest.size() - 1; i++) {
									line(shoulder_detection_image, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], blue, 5, 8, 0);
								}
								goto Exit;

							}
						}
					}
				}
			}
		}

		Exit:
		//Add more point for fail detections
		bool is_intersect = false;
		int j = 2;
		Point2f upper_shoulder_point02 = Point2f(head_upper_shoulder.x + checking_block*j*cos(radian), 
												head_upper_shoulder.y - checking_block*j*sin(radian));
		if (doIntersect(upper_shoulder_point02, symmetric_point, chin, shoulder_line.back())) {
			Point2f intersection_point;
			intersection(upper_shoulder_point02, symmetric_point, chin, shoulder_line.back(), intersection_point);
			line(shoulder_detection_image, intersection_point, shoulder_line.back(), blue, 5, 8, 0);
			shoulder_line.push_back(intersection_point);
		}

		j = 1;
		Point2f upper_shoulder_point01 = Point2f(head_upper_shoulder.x + checking_block*j*cos(radian),
												head_upper_shoulder.y - checking_block*j*sin(radian));
		Point2f above_chin = Point2f(chin.x, chin.y - checking_block);
		circle(shoulder_detection_image, above_chin, 5, green, -1, 8);

		if (doIntersect(upper_shoulder_point01, symmetric_point, above_chin, shoulder_line.back())) {
			Point2f intersection_point;
			intersection(upper_shoulder_point01, symmetric_point, above_chin, shoulder_line.back(), intersection_point);
			line(shoulder_detection_image, intersection_point, shoulder_line.back(), blue, 5, 8, 0);
			shoulder_line.push_back(intersection_point);
		}

		return shoulder_line;
	}
}
void MYcppGui::RefinePoint_collection(Mat& frame, cv::vector<cv::vector<Point2f>> &point_collection) {
	for (int j = 0; j < point_collection.size(); j++) {
		vector<Point2f> point_line = point_collection[j];
		for (int i = 0; i < (int)point_line.size() - 1; i++) {
			if (EuclideanDistance(point_line[i], point_line[i + 1]) > checking_block*1.5) {
				circle(frame, point_line[i], 5, blue, -1, 8);
				circle(frame, point_line[i + 1], 5, blue, -1, 8);

				if (i >= 2) {	//Got more than 2 point backward
					//Remove outside point
					point_line.erase(point_line.begin() + i + 1, point_line.end());

					//Remove some inside points of point_line
					int taken_number = max(0, (int)point_line.size() - max(2, (int)point_line.size() / 3));
					point_line.erase(point_line.begin(), point_line.begin() + taken_number);

					//go out
					break;
				}
			}
		}

		//If there is no outlier
		//Remove some inside points of point_line
		int taken_number = point_line.size() / 2.5;
		point_line.erase(point_line.begin(), point_line.begin() + taken_number);

		//set back
		point_collection[j] = point_line;

		for (int i = 0; i < point_collection[j].size(); i++) {
			circle(frame, point_line[i], 5, green, -1, 8);
		}
	}
}
Mat MYcppGui::Preprocessing(Mat frame) {
	Mat CannyWithoutBlurAndMorphology, bilateralBlur;
	int bilateralIndx = 19;
	bilateralFilter(frame, bilateralBlur, bilateralIndx, bilateralIndx * 2, bilateralIndx / 2);
	CannyProcessing(bilateralBlur, CannyWithoutBlurAndMorphology);

	int erosion_size = 4;
	Mat element = getStructuringElement(cv::MORPH_CROSS,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	cv::dilate(CannyWithoutBlurAndMorphology, CannyWithoutBlurAndMorphology, element);

	int morph_elem = 0;
	//kernel for morphology blur
	int morph_size = 4;
	// Since MORPH_X : 2,3,4,5 and 6
	Mat elementx = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	/// Apply the specified morphology operation
	morphologyEx(CannyWithoutBlurAndMorphology, CannyWithoutBlurAndMorphology, MORPH_CLOSE, elementx);
	return CannyWithoutBlurAndMorphology;
}


void MYcppGui::collectColorShoulderFromInput()
{
	for (int k = 0; k < 2; k++) {	
		for (int i = 0; i < userInput[k].size(); i+=5)
		{
			//Get color inside 25px point
			int x = userInput[k][i].x;
			int y = userInput[k][i].y + 25;
			Vec3b color = userInputFrame.at<Vec3b>(Point2f(x, y));
		
			if (i == 0 && k == 0) 
			{
				colorCollection_Shoulder.push_back(color);
				continue;
			}

			double minColourDistance = ColourDistance(color, colorCollection_Shoulder[0]);
			cout << minColourDistance << endl;

			for (int j = 1; j < colorCollection_Shoulder.size(); j++)
			{
				double colourDistance = ColourDistance(color, colorCollection_Shoulder[j]);
				cout << ColourDistance(color, colorCollection_Shoulder[j]) << endl;
				if (colourDistance < minColourDistance)
					minColourDistance = colourDistance;
			}

			circle(userInputFrame, Point2f(x, y), 5, green, -1, 8);

			if (minColourDistance > 30)
			{
				colorCollection_Shoulder.push_back(color);
				circle(userInputFrame, Point2f(x, y), 5, blue, -1, 8);
			}

			cout << "-----------------" << endl;
		}
	}
}

//void MYcppGui::collectColorShoulder_LAB()
//{
//	Mat LAB_frame;
//	cvtColor(userInputFrame, LAB_frame, CV_BGR2Lab);
//	for (int k = 0; k < 2; k++) {
//		for (int i = 0; i < userInput[k].size(); i += 5)
//		{
//			//Get color inside 25px point
//			int x = userInput[k][i].x;
//			int y = userInput[k][i].y + 25;
//
//			Vec3b color = LAB_frame.at<Vec3b>(Point2f(x, y));
//			//cout << (float)color[0] << ", " << (float)color[1] << ", " << (float)color[2] << endl;
//			cout << (float)color[0] / 255 * 100 << ", " << (float)color[1] - 128 << ", " << (float)color[2] - 128 << endl;
//			Vec3f color_LAB = Vec3f((float)color[0] / 255 * 100, (float)color[1] - 128, (float)color[2] - 128);
//
//			if (i == 0 && k == 0)
//			{
//				colorCollection_LAB.push_back(color_LAB);
//				circle(userInputFrame, Point2f(x, y), 5, blue, -1, 8);
//				continue;
//			}
//
//			double minColourDistance = ColourDistance_LAB(color_LAB, colorCollection_LAB[0]);
//			cout << minColourDistance << endl;
//
//			for (int j = 1; j < colorCollection_LAB.size(); j++)
//			{
//				double colourDistance = ColourDistance_LAB(color_LAB, colorCollection_LAB[j]);
//				cout << ColourDistance_LAB(color_LAB, colorCollection_LAB[j]) << endl;
//				if (colourDistance < minColourDistance)
//					minColourDistance = colourDistance;
//			}
//
//			circle(userInputFrame, Point2f(x, y), 5, green, -1, 8);
//
//			if (minColourDistance > 10)
//			{
//				cout << "Min: " << minColourDistance << endl;
//				colorCollection_LAB.push_back(color_LAB);
//				circle(userInputFrame, Point2f(x, y), 5, blue, -1, 8);
//			}
//
//			cout << "-----------------" << endl;
//
//
//		}
//	}
//}
//double ColourDistance_LAB(Vec3f e1, Vec3f e2)
//{
//	double distance = sqrt((e1[1] - e2[1])*(e1[1] - e2[1]) + (e1[2] - e2[2])*(e1[2] - e2[2]));	//just skipp L, or I can put a bit L later
//	return distance;
//}
//bool MYcppGui::IsMatchToColorCollectionInput_LAB(Vec3f color_LAB)
//{
//
//	for (int i = 0; i < colorCollection_LAB.size(); ++i)
//	{
//		if (ColourDistance_LAB(color_LAB, colorCollection_LAB[i]) <= 10){
//			return true;
//		}
//	}
//	return false;
//}
double ColourDistance(Vec3b e1, Vec3b e2)
{
	//0 - blue 
	//1 - green
	//2 - red
	long rmean = ((long)e1[2] + (long)e2[2]) / 2;
	long r = (long)e1[2] - (long)e2[2];
	long g = (long)e1[1] - (long)e2[1];
	long b = (long)e1[0] - (long)e2[0];
	return sqrt((((512 + rmean)*r*r) >> 8) + 4 * g*g + (((767 - rmean)*b*b) >> 8));
}



bool MYcppGui::IsMatchToColorCollectionInput(Vec3b color)
{

	for (int i = 0; i < colorCollection_Shoulder.size(); ++i)
	{
		if (ColourDistance(color, colorCollection_Shoulder[i]) <= 30){
			return true;
		}
	}
	return false;
}

bool MYcppGui::IsMatchColor(Vec3b color, Vector<Vec3b> Collection, int epsilon)
{
	for (int i = 0; i < Collection.size(); ++i)
	{
		if (ColourDistance(color, Collection[i]) <= epsilon){
			return true;
		}
	}
	return false;
}

//

double EuclideanDistance(Point2f p1, Point2f p2) {
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)); 
}


double Angle(Point2f start, Point2f end)
{
	const double Rad2Deg = 180.0 / CV_PI;
	const double Deg2Rad = CV_PI / 180.0;
	return atan2(start.y - end.y, end.x - start.x) * Rad2Deg;
}

cv::vector<Point2f> MYcppGui::findPath(int index, int index_line, cv::vector<cv::vector<Point2f>> point_collection, double angle) {
	//cout << index << " " << index_line << " " << point_collection.size() << endl;
	if (index_line >= point_collection.size() - 1) {
		cv::vector<Point2f> result;
		result.push_back(point_collection[index_line][index]);
		return result;
	}

	cv::vector<Point2f> new_point_line;
	cv::vector<Point2f> tmp_new_point_line;

	for (int i = point_collection[index_line + 1].size() - 1; i >= 0; i--)		//Go inside out to choose the outter result
																				// I will take all later and pick the closest one to previous result
	{
		//check angle
		if (abs(Angle(point_collection[index_line][index], point_collection[index_line + 1][i]) - angle) <= 30) { //used to 25
			tmp_new_point_line = findPath(i, index_line + 1, point_collection, angle);
		}
		tmp_new_point_line.push_back(point_collection[index_line][index]);

		if (new_point_line.empty() || tmp_new_point_line.size() > new_point_line.size())
		{
			new_point_line = tmp_new_point_line;
		}
		tmp_new_point_line.clear();
	}

	//The case that last chosen point was missing because the next index_line 's size  == 0
	//I'll refactor later
	if (point_collection[index_line + 1].size() == 0) {
		tmp_new_point_line.push_back(point_collection[index_line][index]);

		if (new_point_line.empty() || tmp_new_point_line.size() > new_point_line.size())
		{
			new_point_line = tmp_new_point_line;
		}
		tmp_new_point_line.clear();
	}
	return new_point_line;
}



float FindY_LineEquationThroughTwoPoint(float x_, Point2f p1, Point2f p2)
{
	//equation of a line from two points
	//y - y1 = (y2 - y1) / (x2 - x1) * (x - x1)
	// => y = (y2 - y1) / (x2 - x1) * (x - x1) + y1 

	int x1 = p1.x;
	int y1 = p1.y;
	int x2 = p2.x;
	int y2 = p2.y;
	float y_ = (float)(y2 - y1) / (float)(x2 - x1) * (float)(x_ - x1) + y1;
	return y_;
}

Point2f mirror(Point2f p, Point2f point0, Point2f point1)
{
	double x0 = point0.x;
	double y0 = point0.y;
	double x1 = point1.x;
	double y1 = point1.y;

	double dx, dy, a, b;
	double x2, y2;
	Point2f p1; //reflected point to be returned 

	dx = x1 - x0;
	dy = y1 - y0;

	a = (dx * dx - dy * dy) / (dx * dx + dy*dy);
	b = 2 * dx * dy / (dx*dx + dy*dy);

	x2 = a * (p.x - x0) + b*(p.y - y0) + x0;
	y2 = b * (p.x - x0) - a*(p.y - y0) + y0;

	p1 = Point2f(x2, y2);

	return p1;

}

bool isSegmentsIntersecting(Point2f& p1, Point2f& p2, Point2f& q1, Point2f& q2) {
	return (((q1.x - p1.x)*(p2.y - p1.y) - (q1.y - p1.y)*(p2.x - p1.x))
		* ((q2.x - p1.x)*(p2.y - p1.y) - (q2.y - p1.y)*(p2.x - p1.x)) < 0)
		&&
		(((p1.x - q1.x)*(q2.y - q1.y) - (p1.y - q1.y)*(q2.x - q1.x))
		* ((p2.x - q1.x)*(q2.y - q1.y) - (p2.y - q1.y)*(q2.x - q1.x)) < 0);
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

Mat MYcppGui::GetThumnail(string fileName) {
	
	VideoCapture capture(fileName);
	if (!capture.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return Mat();;
	}

	Mat frame;
	bool bSuccess = capture.read(frame); // read a new frame from video

	if (!bSuccess)
	{
		cout << "Cannot read the frame from video file" << endl;
		return Mat();
	}

	return frame;
}