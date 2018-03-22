#pragma once

#include "cppOpencv.h"

vector<Experiment> Resolutions(3);
int current_px = 0;


//Only Image Version return frame
vector<Mat> MYcppGui::TotalProcess(string fileName, bool image_version, bool saveOnly) {
	Mat frame;
	vector<Mat> frames;
	string pathUserInput = "D:\\605\\Source code\\UserInput\\input_";
	string pathData = "D:\\605\\Source code\\dataset\\train_";
	FILE_NAME = fileName;

	//Correct form input
	for (int i = fileName.size(); i < 3; i++) {
		fileName = "0" + fileName;
	}

	pathUserInput = pathUserInput + fileName + ".txt";
	AddUserInput(pathUserInput);

	//for image
	if (image_version) {
		pathData = pathData + fileName + ".jpg";
		frame = cv::imread(pathData, CV_LOAD_IMAGE_COLOR);
		frames = ImageProcessing_Final(frame, false, true, true);
		return frames;
	}
	//for video version
	else {
		pathData = pathData + "video_" + fileName + ".avi";
		VideoProcessing(pathData, saveOnly);
	}
	return frames;
}

Mat equalizedColor(Mat src) {

	vector<Mat> channels;
	Mat img_hist_equalized;

	cvtColor(src, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format

	split(img_hist_equalized, channels); //split the image into channels

	equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

	merge(channels, img_hist_equalized); //merge 3 channels including the modified 1st channel into one image

	cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

	return img_hist_equalized;
}

double MYcppGui::OverlapPercentage(vector<Point2f> groundTruth, vector<Point2f> points) {
	double NUM_OVERLAP_POINT = 0;
	double NUM_NOT_COUNT = 0;
	Mat TestMat = Mat::zeros(userInputFrame.size(), CV_8UC3);
	TestMat = Scalar::all(0);
	
	for (int i = 0; i < points.size() - 1; i++) {
		line(TestMat, points[i], points[i + 1], white, 20, 8, 0);
	}

	for (int i = 0; i < groundTruth.size(); i++) {
		if (groundTruth[i].y > points[0].y ) {
			circle(TestMat, groundTruth[i], 0.5, red, -1, 8);
			NUM_NOT_COUNT++;
		}
		else {
			Vec3b color = TestMat.at<Vec3b>(groundTruth[i]);
			Scalar cvColor = Scalar(color[0], color[1], color[2]);
			circle(TestMat, groundTruth[i], 0.5, yellow, -1, 8);
			if (cvColor == white) {
				NUM_OVERLAP_POINT++;
				circle(TestMat, groundTruth[i], 0.5, green, -1, 8);
			}
		}
	}
	double percentage = NUM_OVERLAP_POINT / (groundTruth.size() - NUM_NOT_COUNT);

	cout << "Percentage: " << NUM_OVERLAP_POINT << "/" << groundTruth.size() - NUM_NOT_COUNT << " : " << percentage << endl;
	
	percentageOverlapDatas.push_back(percentage);

	return percentage;
}

vector<double> MYcppGui::CompareToGroundTruth(vector<vector<Point2f>> line) {
	vector<double> result;
	for (int k = 0; k < userInput.size() && userInput[k].size() != 0; k++) {
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

bool CheckCommon(std::vector<Point2f> inVectorA, std::vector<Point2f> nVectorB) {
	return std::find_first_of(inVectorA.begin(), inVectorA.end(),
		nVectorB.begin(), nVectorB.end()) != inVectorA.end();
}

Mat Combine2MatSideBySide(Mat &im1, Mat &im2) {
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

Mat ChangeBrightness(Mat image, int beta = 50) {
	double alpha = 1.0;		//Simple contrast control
	//int beta = 50;        //Simple brightness control
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
	merge(result_mask, m1);           // imshow("m1", m1);

	addWeighted(m, 1, m1, 1, 0, m1);    //    imshow("m2", m1);

	m1.copyTo(background);
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
bool onSegment(Point2f p, Point2f q, Point2f r) {
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
int orientation(Point2f p, Point2f q, Point2f r) {
	int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0;  // colinear

	return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool doIntersect(Point2f p1, Point2f q1, Point2f p2, Point2f q2) {
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

MYcppGui::MYcppGui() {
	checking_block = 0;
	dlib::deserialize("D:\\shape_predictor_68_face_landmarks.dat") >> shape_predictor;
}

MYcppGui::~MYcppGui() {
	cvDestroyAllWindows();
}

vector<vector<Point2f>> SimplifizeResult(vector<vector<Point2f>> result) {
	vector<vector<Point2f>> Simplifized;

	Simplifized.resize(result.size());
	for (int k = 0; k < result.size(); k++) {
		if (result[k].size() != 0) {
			approxPolyDP(Mat(result[k]), Simplifized[k], 5, true);		//10
		}
	}
	return Simplifized;
}

vector<vector<Point2f>> MYcppGui::readUserInput(string path) {
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
	}	
	inputs.push_back(leftShoulderInput);
	inputs.push_back(rightShoulderInput);

	return inputs;
}

void MYcppGui::AddUserInput(string path) {
		userInput = readUserInput(path);
		current_shoulderLine = vector<vector<Point2f>>(userInput);
		
		//Simplifized User Input.
		simplifizedUserInput = SimplifizeResult(userInput);
		simplifized_current_shoulderLine = vector<vector<Point2f>>(simplifizedUserInput);
		userInputFrame = NULL;
	

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

int MYcppGui::myCppLoadAndShowRGB(string fileName) {
	cout << fileName << endl;
	fileName = "D:\\train_02.jpg";
	cv::Mat img_input = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
	cv::namedWindow("Source", CV_WINDOW_NORMAL);
	cv::resizeWindow("Source", 282, 502);
	cv::imshow("Source", img_input);
	return 0;
}

void MYcppGui::VideoProcessing(string fileName, bool saveOnly) {
	VIDEO_MODE = true;

	VideoCapture capture(fileName);
	if (!capture.isOpened()) {  // if not success, exit program
	
		cout << "Cannot open the video file" << endl;
		return;
	}
	double fps = capture.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
	bool skip_frame = false;
	if (fps > 35) {
		skip_frame = true;
	}

	cout << "Frame per seconds : " << fps << endl;

	int ratio = 2;
	int w = 1060;
	if (!TEST_MODE) {
		ratio = 1;
		w = 530;
	}
	cv::namedWindow("Source", CV_WINDOW_NORMAL);
	cv::resizeWindow("Source", w, 700);

	//Save the vide0
	int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	
	int str_lenght = fileName.size();
	string noFile = fileName.substr(str_lenght - 7, str_lenght - 5); //dont know why it crop from str_lenght - 6 to the end, so I crop it once again
	noFile = noFile.substr(0, 3);
	
	VideoWriter video("output_" + noFile + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width*ratio, frame_height), true);

	//----------------------Sticker---------------------------
	GetSticker(stickerName, false);

	//-------------------------collect sample color of shouder--------------------
	if (TRACKING_MODE) {
		collectColorShoulderFromInput(userInputFrame);
	}
	
	while (1) {
		cout << nth << " th frame" << endl;
		Mat frame, face_processed;
		bool bSuccess = capture.read(frame); // read a new frame from video

		//skip frame if the video in 60fps
		if (skip_frame && nth % 2 == 0)
		{
			nth++;
			continue;
		}

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			/*capture = VideoCapture(fileName);
			bSuccess = capture.read(frame);*/
		}

		face_processed = frame.clone();
		//Mat tmp_frame = equalizedColor(frame);
		ImageProcessing_Final(face_processed, true, false, true);
		//ImageProcessing_Final(tmp_frame, true, false, true);

		if (TEST_MODE) {
			frame = face_processed;
		}

		if (STICKER_MODE) {
			AddSticker(frame);
		}
		
		if (!saveOnly) cv::imshow("Source", frame);
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

	if (EXPERIMENT_MODE) {
		double size = max(frame.rows, frame.cols);
		if (size < 1000) {
			current_px = 0;
		}
		else if (size < 1500) {
			current_px = 1;
		}
		else {
			current_px = 2;
		}
	}

	//frame = equalizedColor(frame);
	vector<Mat> returnMats;
	Mat src, face_detection_frame; // change name soon
	src = frame.clone(); // Use for Canny
	originalFrame = frame.clone();
	face_detection_frame = frame.clone(); // Use for shoulder detection

	if (userInputFrame.empty()) {
		userInputFrame = frame.clone();
	}

	// Thesis
	//Mat tmp = Preprocessing(src);

	clock_t tmp01;
	if (EXPERIMENT_MODE) { tmp01 = clock(); }

	//------------- Face Detection ------------------
	std::vector<dlib::full_object_detection> shapes_face;
	shapes_face = face_detection_update(face_detection_frame);

	clock_t tmp02 = clock();
	//No face is detected
	if (shapes_face.size() == 0) {
		return returnMats;
	}

	double face_detection_cost;
	if (EXPERIMENT_MODE) {
		Resolutions[current_px].number++;
		face_detection_cost = float(tmp02 - tmp01) / CLOCKS_PER_SEC;
		std::cout << " Time - Whole face detection: " << face_detection_cost << endl;
		Resolutions[current_px].face_detection_cost.push_back(face_detection_cost);
	}

	//testing
	//Get face Line for Masking in AddSticker step
	vector<Point2f> leftFaceLine;
	vector<Point2f> rightFaceLine;
	for (int i = 6; i >= 2; i--) {
		line(face_detection_frame, Point2f(shapes_face[0].part(i).x(), shapes_face[0].part(i).y()),
			Point2f(shapes_face[0].part(i + 1).x(), shapes_face[0].part(i + 1).y()), green, 5, 8, 0);
		line(face_detection_frame, Point2f(shapes_face[0].part(16 - i).x(), shapes_face[0].part(16 - i).y()),
			Point2f(shapes_face[0].part(16 - (i + 1)).x(), shapes_face[0].part(16 - (i + 1)).y()), green, 5, 8, 0);

		leftFaceLine.push_back(Point2f(shapes_face[0].part(i).x(), shapes_face[0].part(i).y()));
		rightFaceLine.push_back(Point2f(shapes_face[0].part(16 - i).x(), shapes_face[0].part(16 - i).y()));
	}
	current_faceLine = vector<vector<Point2f>> {leftFaceLine, rightFaceLine};

	//-----------------------------skin---------------------------
	if (EXPERIMENT_MODE) { tmp01 = clock(); }

	Mat LightUpFrame = ChangeBrightness(src,50);
	
	double color_collection_cost;
	if (EXPERIMENT_MODE) {
		tmp02 = clock();
		color_collection_cost = float(tmp02 - tmp01) / CLOCKS_PER_SEC;
		std::cout << " Time - color_collection_cost: " << color_collection_cost << endl;
		Resolutions[current_px].color_collection_cost.push_back(color_collection_cost);
	}

	Mat skin = GetSkin(src);
	Mat mask_skin;
	cvtColor(skin, mask_skin, CV_BGR2GRAY);
	threshold(mask_skin, mask_skin, 0, 255, THRESH_BINARY);

	//------- Update face // Wrong somethings, 
	CorrectFaceDetection(shapes_face, mask_skin);

	detectNecessaryPointsOfFace(shapes_face);
	circle(face_detection_frame, left_cheek, 5, green, -1, 8);
	circle(face_detection_frame, right_cheek, 5, green, -1, 8);
	circle(face_detection_frame, top_nose, 5, green, -1, 8);
	circle(face_detection_frame, nose, 5, green, -1, 8);
	circle(face_detection_frame, chin, 5, green, -1, 8);
	circle(face_detection_frame, symmetric_point, 5, green, -1, 8);
	circle(face_detection_frame, upper_symmetric_point, 5, green, -1, 8);

	//Thesis
	//circle(userInputFrame, left_cheek, 7, green, -1, 8);
	//circle(userInputFrame, right_cheek, 7, green, -1, 8);
	//circle(userInputFrame, top_nose, 7, green, -1, 8);
	//circle(userInputFrame, chin, 7, green, -1, 8);
	//circle(userInputFrame, symmetric_point, 7, green, -1, 8);

	double left_eye_width = shapes_face[0].part(39).x() - shapes_face[0].part(36).x();
	if (checking_block == 0 || abs(left_eye_width * 1 / 3 - checking_block) >= 50) {
		checking_block = left_eye_width * 1.3 / 2;
	}

	//-----------------------------	Preprocess a part of image to speed up ---------------------------
	if (EXPERIMENT_MODE) { tmp01 = clock(); }

	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x);
	//A ---- B
	//|	     | 
	//D ---- C
	Point2f pA = Point2f(max(left_cheek.x - range_of_shoulder_sample*1.6, 0.0), min(left_cheek.y, right_cheek.y));
	Point2f pB = Point2f(min(right_cheek.x + range_of_shoulder_sample*1.6, double(frame.cols)), pA.y);
	Point2f pC = Point2f(pB.x, min(symmetric_point.y, float(frame.rows)));
	Point2f pD = Point2f(pA.x, pC.y);

	Mat sub_frame = src(cv::Rect(pA.x, pA.y, pB.x - pA.x, pD.y - pA.y));
	Mat CannyWithoutBlurAndMorphology = Preprocessing(sub_frame);

	//Darken Mat for color collection
	Mat DarkenSubFrame = ChangeBrightness(sub_frame, -70);
	Mat DarkenFrame(frame.rows, frame.cols, CV_8UC3, Scalar(0));
	DarkenSubFrame.copyTo(DarkenFrame(cv::Rect(pA.x, pA.y, DarkenSubFrame.cols, DarkenSubFrame.rows)));

	//Thesis
	//Mat CannyWithoutBlurAndMorphology_tmp = Preprocessing(src);

	//Add preprocessed part to a frame that is in the same size with the old one
	Mat BiggerCannyWithoutBlurAndMorphology(frame.rows, frame.cols, CV_8UC1, Scalar(0));
	CannyWithoutBlurAndMorphology.copyTo(BiggerCannyWithoutBlurAndMorphology(cv::Rect(pA.x, pA.y, CannyWithoutBlurAndMorphology.cols, CannyWithoutBlurAndMorphology.rows)));

	//-----------------------------collect Color Shoulder---------------------------
	//Thesis
	//withUserInput = true;

	if (withUserInput) {
		if (nth == 1) {
			collectColorShoulderFromInput(userInputFrame);
			collectColorShoulderFromInput(LightUpFrame);
			collectColorShoulderFromInput(DarkenFrame);
		}
		collectColorShoulder(src, mask_skin, true);
		collectColorShoulder(LightUpFrame, mask_skin, true);
		collectColorShoulder(DarkenFrame, mask_skin, true);

		if (nth % 10 == 0) {
			RefineColorCollection(colorCollection_Shoulder); 
			RefineColorCollection(colorCollection_Skin);
		}
	}
	if (!withUserInput) {
		collectColorShoulder(face_detection_frame, mask_skin, false);
		collectColorShoulder(LightUpFrame, mask_skin, false);
		collectColorShoulder(DarkenFrame, mask_skin, false);
	}

	double preprocess_cost;
	if (EXPERIMENT_MODE) {
		tmp02 = clock();
		preprocess_cost = float(tmp02 - tmp01) / CLOCKS_PER_SEC;
		std::cout << " Time - preprocess_cost: " << preprocess_cost << endl;
		Resolutions[current_px].preprocess_cost.push_back(preprocess_cost);
	}

	//---------------------- STRONGLY BLUR VERSION
	Mat detected_edges;
	Mat Bigger_detected_edges(frame.rows, frame.cols, CV_8UC1, Scalar(0));
	if (isTesting) {
		cv::namedWindow("Erosion After Canny", CV_WINDOW_NORMAL);
		cv::resizeWindow("Erosion After Canny", 282, 502);
		cv::namedWindow("Canny Only", CV_WINDOW_NORMAL);
		cv::resizeWindow("Canny Only", 282, 502);

		//sub_frame = frame(cv::Rect(pA.x, pA.y, pB.x - pA.x, pD.y - pA.y));

		detected_edges = StrongPreprocessing(sub_frame);
		detected_edges.copyTo(Bigger_detected_edges(cv::Rect(pA.x, pA.y, detected_edges.cols, detected_edges.rows)));
		cv::imshow("Erosion After Canny", Bigger_detected_edges);
	}

	//clock_t tmp02 = clock();
	//std::cout << " Time for Preprocess: " << float(tmp02 - tmp01) / CLOCKS_PER_SEC << endl;

	//-----------------------------neck---------------------------
	int angle_neck_left = -100;
	int angle_neck_right = -80;
	vector<Point2f> leftNeckLine = DetectNeckLines(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, mask_skin, shapes_face, true, angle_neck_left);
	vector<Point2f> rightNeckLine = DetectNeckLines(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, mask_skin, shapes_face, false, angle_neck_right);
	
	//Improve Neck line
	leftNeckLine.push_back(Point2f(leftNeckLine.back().x, leftNeckLine.back().y - checking_block / 2));
	leftNeckLine.insert(leftNeckLine.begin(), Point2f(leftNeckLine[0].x, leftNeckLine[0].y + checking_block / 2));

	rightNeckLine.push_back(Point2f(rightNeckLine.back().x, rightNeckLine.back().y - checking_block / 2));
	rightNeckLine.insert(rightNeckLine.begin(), Point2f(rightNeckLine[0].x, rightNeckLine[0].y + checking_block / 2));

	current_neckLine = vector<vector<Point2f>> {leftNeckLine, rightNeckLine};

	//-----------------------------shoulders------------------	---------
	Mat face_detection_frame_Blur_NoCheck = face_detection_frame.clone();

	if (EXPERIMENT_MODE) { tmp01 = clock(); }

	vector<Point2f> leftShouderLine = detectShoulderLine(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, 
		true, LEFT_SHOULDER_ANGLE, green, true, false); // isTesting
	vector<Point2f> rightShouderLine = detectShoulderLine(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, 
		false, RIGHT_SHOULDER_ANGLE, green, true, false); //isTesting

	if (EXPERIMENT_MODE) {
		tmp02 = clock();
		double shoulder_detection_cost = float(tmp02 - tmp01) / CLOCKS_PER_SEC;
		std::cout << " Time - shoulder_detection_cost: " << shoulder_detection_cost << endl;
		Resolutions[current_px].shoulder_detection_cost.push_back(shoulder_detection_cost);

		double total_cost = face_detection_cost + preprocess_cost + color_collection_cost + shoulder_detection_cost;
		Resolutions[current_px].total_cost.push_back(total_cost);
	}

	current_shoulderLine = vector<vector<Point2f>> {leftShouderLine, rightShouderLine};
	simplifized_current_shoulderLine = SimplifizeResult(current_shoulderLine);

	//Compare result to ground truth
	if (EXPERIMENT_MODE) {
		vector<double> percentages = CompareToGroundTruth(current_shoulderLine);
	}

	//-----------------------------testing shoulder---------------------------
	if (isTesting) {
		detectShoulderLine(face_detection_frame_Blur_NoCheck, Bigger_detected_edges, true, LEFT_SHOULDER_ANGLE, blue, false, false);
		detectShoulderLine(face_detection_frame_Blur_NoCheck, Bigger_detected_edges, false, RIGHT_SHOULDER_ANGLE, blue, false, false);

		cv::namedWindow("Blur_NoCheck", CV_WINDOW_NORMAL);
		cv::resizeWindow("Blur_NoCheck", 530, 700);

		cv::imshow("Blur_NoCheck", face_detection_frame_Blur_NoCheck);

		cv::namedWindow("Source_NoBlur_Check", CV_WINDOW_NORMAL);
		cv::resizeWindow("Source_NoBlur_Check", 530, 700);
		cv::imshow("Source_NoBlur_Check", face_detection_frame);
		cv::imshow("Canny Only", BiggerCannyWithoutBlurAndMorphology);

		//Combine2MatSideBySide
		cvtColor(Bigger_detected_edges, Bigger_detected_edges, CV_GRAY2RGB);
		Mat combine = Combine2MatSideBySide(face_detection_frame_Blur_NoCheck, Bigger_detected_edges);
		returnMats.push_back(combine);
	}

	//Return face_detection_frame
	if (DebugLine) {
		frame = face_detection_frame.clone();
	}
	else {
		frame = src.clone();
	}

	cvtColor(BiggerCannyWithoutBlurAndMorphology, BiggerCannyWithoutBlurAndMorphology, CV_GRAY2RGB);
	Mat combine = Combine2MatSideBySide(frame, BiggerCannyWithoutBlurAndMorphology);
	frame = combine.clone();
	returnMats.push_back(frame);
	std::cout << " Time for Postprocess: " << float(clock() - tmp02) / CLOCKS_PER_SEC << endl << endl;
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
				stickerStatus = Disappearing;
			}
		}

		// Appearing
		if (x_ROI_sticker_begin >= (stickerWidth - distanceMoving) && stickerStatus == Disappearing) {
			stickerStatus = Appearing;
			x_ROI_sticker_begin = 0;	//use as stickerWidth when Appearing
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
				stickerPosition = Point2f(right_neck, current_shoulderLine[RIGHT_LINE].back().y - checking_block/1);	//+ checking_block/2 for go high abit
			}
			if (stickerDirection == RIGHT) {
				stickerPosition = Point2f(left_neck - (stickerWidth - x_ROI_sticker_begin), current_shoulderLine[LEFT_LINE].back().y - checking_block / 1);
			}
		}

		if (stickerStatus == Appearing) {
			//Update x_ROI_sticker_begin
			x_ROI_sticker_begin += distanceMoving; 
			need_to_crop_sticker = true;
			if (stickerDirection == LEFT) {
				stickerPosition = Point2f(left_neck - x_ROI_sticker_begin, current_shoulderLine[LEFT_LINE].back().y - checking_block / 1);
			}
			if (stickerDirection == RIGHT) {
				stickerPosition = Point2f(right_neck, current_shoulderLine[RIGHT_LINE].back().y - checking_block / 1);
			}
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

		//Crop the sticker
		if (need_to_crop_sticker) {
			// Setup a rectangle to define your region of interest
			double ratio = sticker.size().width / stickerWidth;		// > 1

			if (stickerStatus == Disappearing) {
				stickerWidth -= x_ROI_sticker_begin;	//Used for croping BG later
				cv::Rect stickerROI(x_ROI_sticker_begin * ratio, 0, sticker.size().width - x_ROI_sticker_begin * ratio, sticker.size().height);

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
		if (((stickerPosition.x <= 0 || stickerPosition.x + stickerWidth / 2 <= current_shoulderLine[LEFT_LINE][0].x) 
			&& stickerDirection == LEFT)
			|| ((stickerPosition.x + stickerWidth >= frame.size().width 
				|| stickerPosition.x + stickerWidth / 2 >= current_shoulderLine[RIGHT_LINE][0].x) 
				&& stickerDirection == RIGHT)) {
			stickerDirection = -1 * stickerDirection; // change direction
			actualPostion_sticker = stickerPosition.x + stickerDirection * 5;		// 1 is in case
			relativePostion_sticker = actualPostion_sticker - nose.x;
			Disappeared = false;
			return;
		}

		//Add sticker
		Mat cropedBG;
		Rect roi(stickerPosition.x, stickerPosition.y - stickerHeight / 2, stickerWidth, stickerHeight);	
		//frame(roi).copyTo(cropedBG);
		originalFrame(roi).copyTo(cropedBG);
		Mat cloneCropedBG = cropedBG.clone();
		Mat dst = overlrayImage(cropedBG, sticker);
		dst.copyTo(frame(roi));

		//Try to add Mask overlay
		if (stickerStatus == Appearing || stickerStatus == Disappearing) {
			Mat maskNeck, maskFace, maskAll, mask2;
			maskNeck = BuildNeckMask(current_neckLine[0], current_neckLine[1]);
			maskFace = BuildNeckMask(current_faceLine[0], current_faceLine[1]);
			bitwise_or(maskNeck, maskFace, maskAll); 

			maskAll(roi).copyTo(mask2);

			overlayMask(dst, cloneCropedBG, mask2);
			dst.copyTo(frame(roi));
		}
		for (int k = 0; k <= 1; k++) {
			for (int i = 0; i < current_neckLine[k].size() - 1; i++) {
				line(frame, current_neckLine[k][i], current_neckLine[k][i+1], green, 3, 8, 0);
			}
		}

}

Mat MYcppGui::ImageProcessing(string fileName, vector<cv::Point2f> userInput) {
	cv::Mat img_input = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
	ImageProcessing_Final(img_input, false, true, true);
	return img_input;
}


void MYcppGui::Morphology_Operations(Mat &src) {
	int morph_elem = 0;

	//kernel for morphology blur
	int morph_size = 10;

	// Since MORPH_X : 2,3,4,5 and 6
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

	/// Apply the specified morphology operation
	morphologyEx(src, src, MORPH_OPEN, element);	//open
	morphologyEx(src, src, MORPH_CLOSE, element);	//close
}

void MYcppGui::CannyProcessing(Mat image, OutputArray edges) {
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
}

std::vector<dlib::full_object_detection> MYcppGui::face_detection_update(Mat frame) {
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

	cout << "Faces detected: " << cur_dets.size() << endl;

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

void MYcppGui::CorrectFaceDetection(std::vector<dlib::full_object_detection>& shapes_face, Mat &mask_skin) {
	//Cheating
	if (FILE_NAME != "002" && FILE_NAME != "023" && FILE_NAME != "047") {
		return;
	}

	Point2f left_point = Point2f(shapes_face[0].part(3).x(), shapes_face[0].part(3).y());
	Point2f right_point = Point2f(shapes_face[0].part(13).x(), shapes_face[0].part(13).y());
	Point2f centre_point = Point2f(shapes_face[0].part(33).x(), shapes_face[0].part(33).y());
	Point2f chin = Point2f(shapes_face[0].part(8).x(), shapes_face[0].part(8).y());
	Point2f top_nose = Point2f(shapes_face[0].part(27).x(), shapes_face[0].part(27).y());

	double distance_3_33 = EuclideanDistance(left_point, centre_point);
	double distance_13_33 = EuclideanDistance(right_point, centre_point);
	bool fix_left;


	//left wrong
	int COUNT = 0;
	if (distance_3_33 - distance_13_33 > checking_block / 3) {		//at first checking_block/3
		for (int i = 1; i <= 5; i++) {
			int color_skin = mask_skin.at<uchar>(Point2f(shapes_face[0].part(i).x() + checking_block / 3, shapes_face[0].part(i).y()));
			if (color_skin == 0) {
				COUNT++;
				circle(userInputFrame, Point2f(shapes_face[0].part(i).x(), shapes_face[0].part(i).y()), 1, green, -1, 8);
			}
		}
		if (COUNT > 2) {
			fix_left = true;
		}
		else return;
	}
	else if (distance_13_33 - distance_3_33 > checking_block / 3) {
		for (int i = 11; i < 16; i++) {
			int color_skin = mask_skin.at<uchar>(Point2f(shapes_face[0].part(i).x() - checking_block / 3, shapes_face[0].part(i).y()));
			if (color_skin == 0) {
				COUNT++;
				circle(userInputFrame, Point2f(shapes_face[0].part(i).x(), shapes_face[0].part(i).y()), 1, green, -1, 8);
			}
		}
		if (COUNT > 2) {
			fix_left = false;
		}
		else return;
		
	}
	else return;
	


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

	double radian_left = LEFT_SHOULDER_ANGLE * CV_PI / 180.0;
	double radian_right = RIGHT_SHOULDER_ANGLE * CV_PI / 180.0;
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

void MYcppGui::RefineColorCollection(vector<Vec3b> &colorCollection) {
	if (colorCollection.size() != 0) {
		vector<Vec3b> newCollection;
		newCollection.push_back(colorCollection[0]);

		for (int i = 1; i < colorCollection.size(); i++) {
			bool isMatchColor = IsMatchColor(colorCollection[i], newCollection, 20);

			if (!isMatchColor) {
				newCollection.push_back(colorCollection[i]);
			}
		}
		colorCollection = newCollection;
	}
}

void MYcppGui::collectColor(Mat&frame, Mat &mask_skin, vector<Vec3b> &colorCollection, Point2f head_point, Point2f end_point, double epsilon, bool splitSkinColor) {
	//LineIterator 
	LineIterator it(frame, head_point, end_point, 8, false);

	for (int i = 0; i < it.count; i += 1, ++it) {
		if (i % 10 != 0) {	//move 10px
			continue;
		}
		int x = it.pos().x;
		int y = it.pos().y;

		Vec3b color = AverageColor(frame, x, y);

		int check_skin = mask_skin.at<uchar>(Point2f(x, y));


		if (splitSkinColor && check_skin == 255) {
			if (colorCollection_Skin.size() == 0) {
				colorCollection_Skin.push_back(color);
				continue;
			}

			double minColourDistance = ColourDistance(color, colorCollection_Skin[0]);

			for (int j = 1; j < colorCollection_Skin.size(); j++) {
				double colourDistance = ColourDistance(color, colorCollection_Skin[j]);
				if (colourDistance < minColourDistance)
					minColourDistance = colourDistance;
			}

			circle(frame, Point2f(x, y), 1, green, -1, 8);		//1

			if (minColourDistance > epsilon) {
				colorCollection_Skin.push_back(color);
				circle(frame, Point2f(x, y), 3, blue, -1, 8);	//3
			}
		}
		else {
			if (colorCollection.size() == 0) {
				colorCollection.push_back(color);
				continue;
			}

			double minColourDistance = ColourDistance(color, colorCollection[0]);
			//cout << minColourDistance << endl;

			for (int j = 1; j < colorCollection.size(); j++) {
				double colourDistance = ColourDistance(color, colorCollection[j]);
				//cout << colourDistance << endl;
				if (colourDistance < minColourDistance)
					minColourDistance = colourDistance;
			}

			circle(frame, Point2f(x, y), 1, green, -1, 8);		//1

			if (minColourDistance > epsilon) {
				colorCollection.push_back(color);
				circle(frame, Point2f(x, y), 3, blue, -1, 8);	//3
			}
		}
	}
}

void MYcppGui::collectColorShoulder(Mat& frame, Mat &mask_skin, bool splitSkinColor) {

	double radian_left = LEFT_SHOULDER_ANGLE * CV_PI / 180.0;
	double radian_right = RIGHT_SHOULDER_ANGLE * CV_PI / 180.0;
	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x); //used to be *2
	double length = abs(range_of_shoulder_sample / cos(radian_left));

	Point2f head_bottom_shoulder = chin;
	Point2f end_bottom_shoulder_left = Point2f(head_bottom_shoulder.x + length*cos(radian_left) + 10, head_bottom_shoulder.y - length*sin(radian_left));
	Point2f end_second_bottom_shoulder_left = Point2f(end_bottom_shoulder_left.x, symmetric_point.y);
	Point2f end_bottom_shoulder_right = Point2f(head_bottom_shoulder.x + length*cos(radian_right) - 10, head_bottom_shoulder.y - length*sin(radian_right));
	Point2f end_second_bottom_shoulder_right = Point2f(end_bottom_shoulder_right.x, symmetric_point.y);

	// Collect color of shoulder 
	if (VIDEO_MODE) {
		Point2f middle_left_bottom_shoulder = Point2f(head_bottom_shoulder.x + length / 2 * cos(radian_left) + 10, head_bottom_shoulder.y - length / 2 * sin(radian_left));;
		Point2f middle_right_bottom_shoulder = Point2f(head_bottom_shoulder.x + length / 2 * cos(radian_right) - 10, head_bottom_shoulder.y - length / 2 * sin(radian_right));;
		collectColor(frame, mask_skin, colorCollection_Shoulder, middle_left_bottom_shoulder, end_bottom_shoulder_left, 20, splitSkinColor);
		collectColor(frame, mask_skin, colorCollection_Shoulder, end_bottom_shoulder_left, end_second_bottom_shoulder_left, 20, splitSkinColor);
		collectColor(frame, mask_skin, colorCollection_Shoulder, middle_right_bottom_shoulder, end_bottom_shoulder_right, 20, splitSkinColor);
		collectColor(frame, mask_skin, colorCollection_Shoulder, end_bottom_shoulder_right, end_second_bottom_shoulder_right, 20, splitSkinColor);
	}
	else {
		//move these point a bit to not effect edge detection(10px)
		for (int i = -1; i <= 1; i += 2) {
			Point2f head_bottom_shoulder = Point2f(chin.x + i*checking_block, chin.y + checking_block);
			Point2f end_bottom_shoulder_left = Point2f(head_bottom_shoulder.x + length*cos(radian_left) + 10, head_bottom_shoulder.y - length*sin(radian_left));
			Point2f end_second_bottom_shoulder_left = Point2f(end_bottom_shoulder_left.x, symmetric_point.y);
			Point2f end_bottom_shoulder_right = Point2f(head_bottom_shoulder.x + length*cos(radian_right) - 10, head_bottom_shoulder.y - length*sin(radian_right));
			Point2f end_second_bottom_shoulder_right = Point2f(end_bottom_shoulder_right.x, symmetric_point.y);

			// Collect color of shoulder 
			collectColor(frame, mask_skin, colorCollection_Shoulder, head_bottom_shoulder, end_bottom_shoulder_left, 20, splitSkinColor);
			collectColor(frame, mask_skin, colorCollection_Shoulder, end_bottom_shoulder_left, end_second_bottom_shoulder_left, 20, splitSkinColor);

			collectColor(frame, mask_skin, colorCollection_Shoulder, head_bottom_shoulder, end_bottom_shoulder_right, 20, splitSkinColor);
			collectColor(frame, mask_skin, colorCollection_Shoulder, end_bottom_shoulder_right, end_second_bottom_shoulder_right, 20, splitSkinColor);

			//Thesis 
			/*collectColor(userInputFrame, mask_skin, colorCollection_Shoulder, head_bottom_shoulder, end_bottom_shoulder_left, 30, splitSkinColor);
			collectColor(userInputFrame, mask_skin, colorCollection_Shoulder, end_bottom_shoulder_left, end_second_bottom_shoulder_left, 30, splitSkinColor);

			collectColor(userInputFrame, mask_skin, colorCollection_Shoulder, head_bottom_shoulder, end_bottom_shoulder_right, 30, splitSkinColor);
			collectColor(userInputFrame, mask_skin, colorCollection_Shoulder, end_bottom_shoulder_right, end_second_bottom_shoulder_right, 30, splitSkinColor);
			*/
		}
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

		if (i == 0) {
			colorNeckCollection.push_back(color);
			continue;
		}

		double minColourDistance = ColourDistance(color, colorNeckCollection[0]);
		cout << minColourDistance << endl;

		for (int j = 1; j < colorNeckCollection.size(); j++) {
			double colourDistance = ColourDistance(color, colorNeckCollection[j]);
			cout << colourDistance << endl;
			if (colourDistance < minColourDistance)
				minColourDistance = colourDistance;
		}

		circle(frame, Point2f(x, y), 5, green, -1, 8);

		if (minColourDistance > 20) {
			colorNeckCollection.push_back(color);
			circle(frame, Point2f(x, y), 5, blue, -1, 8);
		}

		cout << "-----------------" << endl;
	}
	return colorNeckCollection;
}

Mat MYcppGui::BuildNeckMask(vector<Point2f>&leftNeckLine, vector<Point2f>&rightNeckLine) {
	Mat mask = Mat::zeros(originalFrame.size(), CV_8UC1);
	vector<Point> NeckLine(leftNeckLine.begin(), leftNeckLine.end());
	NeckLine.insert(NeckLine.end(), rightNeckLine.rbegin(), rightNeckLine.rend());

	vector <vector<Point> > contourElement;
	contourElement.push_back(NeckLine);
	vector<Point> tmp = contourElement.at(0);
	const Point* elementPoints[1] = { &tmp[0] };
	int numberOfPoints = (int)tmp.size();
	fillPoly(mask, elementPoints, &numberOfPoints, 1, Scalar(255), 8);
	return mask;
}

vector<Point2f> MYcppGui::DetectNeckLines(Mat shoulder_detection_image, Mat detected_edges, Mat mask_skin, std::vector<dlib::full_object_detection> shapes_face, bool leftHandSide, int angle_neck) {
	
	Point2f head_neck;
	Point2f end_neck;
	Point2f head_neck02;
	Point2f end_neck02;
	Vector<Vec3b> colors;

	if (leftHandSide) {
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
	vector<vector<Point2f>> point_collection;

	for (int i = 0; head_neck.y + i*checking_block / 2 < end_neck.y; i++) {
		double Y = head_neck.y + i*checking_block / 2;
		vector<Point2f> point_line;

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
		for (int b1 = 0; b1 < point_collection[a].size(); b1++) {
			for (int b2 = 0; b2 < point_collection[a + 1].size(); b2++) {
				//Check difference of angle
				if (AngleDifference(Angle(point_collection[a][b1], point_collection[a + 1][b2]), angle_neck) <= 15) {
					line(shoulder_detection_image, point_collection[a][b1], point_collection[a + 1][b2], yellow, 2, 8, 0);
				}
			}
		}
	}
	
	vector<vector<Point2f>> possible_lines;
	vector<Point2f> neck_line;
	for (int i = 0; i < point_collection.size(); i++) {
		for (int j = 0; j < point_collection[i].size(); j++) {
			vector<Point2f> path = findPath(i, j, point_collection, angle_neck, 25, 2);

			if (possible_lines.empty() || path.size() > possible_lines.back().size()) {
				if (!path.empty()) {
					possible_lines.push_back(path);
				}
			}
		}
	}

	if (!possible_lines.empty()) {
		neck_line = possible_lines.back();

		for (int i = 0; i < neck_line.size() - 1; i++) {
			line(shoulder_detection_image, neck_line[i], neck_line[i + 1], green, 3, 8, 0);
		}	
	}
	return neck_line;
}

vector<vector<Point2f>> MYcppGui::Collect_Potential_ShoulderPoint(Mat shoulder_detection_image, Mat &detected_edges, ShoulderModel &shoulderModel, bool checkColor, bool checkPreviousResult, Scalar color){

	vector<vector<Point2f>> point_collection;

	//intersection_point_01 to mark where the second bottom shoulder line start
	Point2f intersection_point_01;
	intersection(shoulderModel.head_upper_shoulder, shoulderModel.end_upper_shoulder, symmetric_point, shoulderModel.end_bottom_shoulder, intersection_point_01);


	//Count points added
	int COUNT_ALL = 0;

	Point2f intersection_point_with_previous_result;

	//Take points on shoulder_sample follow "checking_block" and build LineIterator from these point to symmetric_point (but stop at bottom shoulder_line
	for (int j = 0; abs(checking_block*j*cos(shoulderModel.radian)) < shoulderModel.range_of_shoulder_sample*2.5; j++) {
		Point2f current_point = Point2f(shoulderModel.head_upper_shoulder.x + checking_block*j*cos(shoulderModel.radian), shoulderModel.head_upper_shoulder.y - checking_block*j*sin(shoulderModel.radian));
		int value_in_edge_map = 0;
		Point2f intersection_point;

		//Find intersection_point which is point to stop the LineIterator ==> Create LineIterator fo
		if (current_point.y <= intersection_point_01.y) {
			intersection(shoulderModel.head_bottom_shoulder, shoulderModel.end_bottom_shoulder, current_point, symmetric_point, intersection_point);
		}
		else {
			intersection(shoulderModel.end_bottom_shoulder, shoulderModel.end_second_bottom_shoulder, current_point, symmetric_point, intersection_point);
		}

		//Got error when run train_23. So i let it stop when touch end_second_bottom_shoulder
		if (intersection_point.y > shoulderModel.end_second_bottom_shoulder.y) {
			break;
		}

		vector<Point2f> point_line;

		// Thesis
		//line(userInputFrame, intersection_point, current_point, blue, 3, 8, 0);

		//LineIterator from  intersection_point which we found above to upper shoulder line 		//Go inside out
		LineIterator it(shoulder_detection_image, intersection_point, current_point, 8, false);

		//Get all intersections of LineIterator and Canny lines;
		for (int i = 0; i < it.count; i += 2, ++it, ++it) {
			// Skip the point out of the frame
			if (it.pos().x < 0 || it.pos().x > shoulder_detection_image.size().width - 1 || it.pos().y > shoulder_detection_image.size().height - 1) {
				continue;
			}

			value_in_edge_map = detected_edges.at<uchar>(it.pos().y, it.pos().x);	// y first, x later
			Point2f current_point = Point2f(it.pos().x, it.pos().y);

			//problem only on train_10
			if (it.pos().y + 15 > shoulder_detection_image.size().height - 1)
				break;

			Vec3b color_inside = AverageColor(originalFrame, it.pos().x, it.pos().y + 18);

			bool is_match_color_shoulder = IsMatchColor(color_inside, colorCollection_Shoulder, 30);
			bool is_match_color = is_match_color_shoulder;

			if (j <= 4){ //Assume as range of neck
				bool is_match_color_neck = IsMatchColor(color_inside, colorCollection_Skin, 30);
				is_match_color = is_match_color_shoulder || is_match_color_neck;
			}

			if (!checkColor) {
				is_match_color = true;
				//is_match_color = IsMatchColor(color_inside, colorCollection_Shoulder, 60);
			}

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

			//Thesis
			//is_match_color = true;

			if (value_in_edge_map == 255 && is_match_color) {
				if (point_line.empty() || EuclideanDistance(current_point, point_line.back()) >= checking_block / 2.5) {	//10 work really well - 15 works well too
				
					circle(shoulder_detection_image, Point2f(it.pos().x, it.pos().y), 2, color, -1, 8);

					//Thesis
					//circle(userInputFrame, Point2f(it.pos().x, it.pos().y), 5, red, -1, 8);

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
	return point_collection;
}
//new fuction
vector<Point2f> MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, bool leftHandSide, int angle, Scalar color, bool checkColor, bool checkPreviousResult)
{
	/*Mat LAB_frame;
	cvtColor(shoulder_detection_image, LAB_frame, CV_BGR2Lab);*/

	double radian = angle * CV_PI / 180.0;
	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x); //used to be *2
	double length = abs(range_of_shoulder_sample / cos(radian));
	
	//bottom shoulder sample line
	Point2f head_bottom_shoulder = Point2f(chin.x, chin.y + checking_block);
	Point2f end_bottom_shoulder = Point2f(head_bottom_shoulder.x + length*cos(radian), head_bottom_shoulder.y - length*sin(radian));
	Point2f end_second_bottom_shoulder = Point2f(end_bottom_shoulder.x, symmetric_point.y);
	
	//upper shoulder sample line
	Point2f head_upper_shoulder;

	if (leftHandSide) {	//GO LEFT
		head_upper_shoulder = Point2f(left_cheek.x - 10, left_cheek.y - 10);	// - distance_from_face_to_shouldersample
	}
	else {
		head_upper_shoulder = Point2f(right_cheek.x + 10, right_cheek.y - 10);	// + distance_from_face_to_shouldersample
	}
	
	Point2f end_upper_shoulder = Point2f(head_upper_shoulder.x + length*2.5*cos(radian), head_upper_shoulder.y - length*2.5*sin(radian));

	//Shoulder Model
	ShoulderModel shoulderModel(angle, radian, range_of_shoulder_sample, length, head_bottom_shoulder, end_bottom_shoulder, end_second_bottom_shoulder, head_upper_shoulder, end_upper_shoulder, leftHandSide);

	// Draw DebugLines
	line(shoulder_detection_image, head_bottom_shoulder, end_bottom_shoulder, red, 5, 8, 0);		//2
	line(shoulder_detection_image, end_bottom_shoulder, end_second_bottom_shoulder, red, 5, 8, 0);
	line(shoulder_detection_image, head_upper_shoulder, end_upper_shoulder, red, 5, 8, 0);

	// Thesis
	//line(userInputFrame, head_bottom_shoulder, end_bottom_shoulder, red, 5, 8, 0);		//2
	//line(userInputFrame, end_bottom_shoulder, end_second_bottom_shoulder, red, 5, 8, 0);
	//line(userInputFrame, head_upper_shoulder, end_upper_shoulder, red, 5, 8, 0);

	vector<vector<Point2f>> point_collection = Collect_Potential_ShoulderPoint(shoulder_detection_image, detected_edges, shoulderModel, checkColor, checkPreviousResult, red);
	
	int COUNT_ALL = 0;
	for (int i = 0; i < point_collection.size(); i++) {
		COUNT_ALL += point_collection[i].size();
	}

	double epsilon_angle_shoulder = 30;
	//
	cout << "COUNT_SHOULDER:";
	//Reduce point_collection in the case that shirt texture is too complicated
	if (COUNT_ALL > 250) {
		RefinePoint_collection_v23(shoulder_detection_image, point_collection, shoulderModel);
	}
	else if (COUNT_ALL > 95) {		
		cout << COUNT_ALL << " --> ";

		//RefinePoint_collection(shoulder_detection_image, point_collection);
		int refine_type = RefinePoint_collection_v2(shoulder_detection_image, detected_edges, point_collection, shoulderModel);

		if (refine_type == 1) {
			epsilon_angle_shoulder = 40;
		}

		//check COUNT_ALL again
		COUNT_ALL = 0;
		for (int j = 0; j < point_collection.size(); j++) {
			COUNT_ALL += point_collection[j].size();
		}
	}


	cout << "	" << COUNT_ALL << endl;
	//THESIS
	//Draw simplifizedUserInput for testing
	//for (int i = 0; i < userInput[!leftHandSide].size(); i++) {
	//	//circle(shoulder_detection_image, userInput[!leftHandSide][i], 5, yellow, -1, 8);
	//	if (i < userInput[!leftHandSide].size() - 1) {
	//		line(userInputFrame, userInput[!leftHandSide][i], userInput[!leftHandSide][i + 1], blue, 3, 8, 0);
	//	}
	//}

	//Draw simplifized_current_shoulderLine for testing
	if (checkPreviousResult || simplifized_current_shoulderLine.size() > 0) {
		for (int i = 0; i < simplifized_current_shoulderLine[!leftHandSide].size(); i++) {
			circle(shoulder_detection_image, simplifized_current_shoulderLine[!leftHandSide][i], 5, yellow, -1, 8);
			if (i < simplifized_current_shoulderLine[!leftHandSide].size() - 1) {
				//line(shoulder_detection_image, simplifized_current_shoulderLine[!leftHandSide][i], simplifized_current_shoulderLine[!leftHandSide][i + 1], yellow, 3, 8, 0);

				//Thesis
				line(userInputFrame, simplifized_current_shoulderLine[!leftHandSide][i], simplifized_current_shoulderLine[!leftHandSide][i + 1], yellow, 3, 8, 0);
			}
		}
	}

	if (nth == 328) {
		cout << endl;
	}

	//take potential point for shoulder line by checking angle of these line
	vector<Point2f> shoulder_line;
	shoulder_line = Finding_ShoulderLines_From_PointCollection_v2(shoulder_detection_image, point_collection, shoulderModel, color, epsilon_angle_shoulder);
	
	if (checkPreviousResult || simplifized_current_shoulderLine.size() > 0) {
		////Thesis
		//Point2f pointA = Point2f(max((float)0, simplifized_current_shoulderLine[!leftHandSide][0].x - 30), simplifized_current_shoulderLine[!leftHandSide][0].y);
		//Point2f pointB = Point2f(min((float)userInputFrame.size().width, simplifized_current_shoulderLine[!leftHandSide][0].x + 30), simplifized_current_shoulderLine[!leftHandSide][0].y);
		//line(userInputFrame, pointA, pointB, yellow, 3, 8, 0);

		//pointA = Point2f(max((float)0, simplifized_current_shoulderLine[!leftHandSide].back().x - 30), simplifized_current_shoulderLine[!leftHandSide].back().y);
		//pointB = Point2f(min((float)userInputFrame.size().width, simplifized_current_shoulderLine[!leftHandSide].back().x + 30), simplifized_current_shoulderLine[!leftHandSide].back().y);
		//line(userInputFrame, pointA, pointB, yellow, 3, 8, 0);
	}
	return shoulder_line;
}

vector<Point2f> MYcppGui::Finding_ShoulderLines_From_PointCollection(Mat shoulder_detection_image, vector<vector<Point2f>> point_collection, ShoulderModel &shoulderModel, Scalar color, double epsilon_angle_shoulder) {

	int angle_for_arm = RIGHT_ARM_ANGLE;
	if (shoulderModel.leftHandSide)
		angle_for_arm = LEFT_ARM_ANGLE;
	
	//Show possible point
	for (int a = 0; a < point_collection.size() - 1; a++) {
		for (int b1 = 0; b1 < point_collection[a].size(); b1++) {
			for (int b2 = 0; b2 < point_collection[a + 1].size(); b2++) {
				//Check difference of angle
				if (AngleDifference(Angle(point_collection[a][b1], point_collection[a + 1][b2]), shoulderModel.angle) <= epsilon_angle_shoulder
					|| AngleDifference(Angle(point_collection[a][b1], point_collection[a + 1][b2]), angle_for_arm) <= 30) {
					if (EuclideanDistance(point_collection[a][b1], point_collection[a + 1][b2]) < checking_block * 1.5) {
						line(shoulder_detection_image, point_collection[a][b1], point_collection[a + 1][b2], red, 3, 8, 0);

						//THESIS
						//line(userInputFrame, point_collection[a][b1], point_collection[a + 1][b2], blue, 2, 8, 0);
					}
					
				}
			}
		}
	}
	vector<vector<Point2f>> possible_lines;
	vector<vector<Point2f>> possible_lines_for_arm;

	for (int i = 0; i < point_collection.size(); i++) {
		for (int j = 0; j < point_collection[i].size(); j++) {
			vector<Point2f> path = findPath(i, j, point_collection, shoulderModel.angle, epsilon_angle_shoulder, 0);
			vector<Point2f> path_for_arm = findPath(i, j, point_collection, angle_for_arm, 30, 1);

			if (possible_lines.empty() || path.size() > possible_lines.back().size()) {
				if (!path.empty()) {
					possible_lines.push_back(path);
				}
			}

			if (possible_lines_for_arm.empty() || path_for_arm.size() > possible_lines_for_arm.back().size()) {
				if (!path_for_arm.empty()) {
					possible_lines_for_arm.push_back(path_for_arm);
					//cout << "A new max line" << endl;
				}
			}
		}
	}

	//old way use the longest one
	vector<Point2f> shoulder_line_for_arm_longest;
	if (!possible_lines_for_arm.empty()) {
		shoulder_line_for_arm_longest = possible_lines_for_arm.back();

		for (int i = 0; i < shoulder_line_for_arm_longest.size() - 1; i++) {
			line(shoulder_detection_image, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], black, 5, 8, 0);

			//THESIS
			//line(userInputFrame, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], green, 5, 8, 0);
		}
	}

	if (!possible_lines.empty()) {
		vector<Point2f> shoulder_line = possible_lines.back();

		for (int i = 0; i < shoulder_line.size() - 1; i++) {
			line(shoulder_detection_image, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);

			// THESIS
			//line(userInputFrame, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);
		}


		//new way which is to detect the position of arm line
		//list use pushback, so the last one is [0]
		int index_one_third = shoulder_line.size() * 1 / 3;
		int index_half = shoulder_line.size() / 2;
		vector<Point2f> shoulder_line_for_arm_test;

		for (int i = possible_lines_for_arm.size() - 1; i >= 0; i--) {
			if (abs(shoulder_line[0].x - possible_lines_for_arm[i].back().x) < abs(shoulder_line[0].x - shoulder_line[index_one_third].x)) {
				if (possible_lines_for_arm[i].back().y > shoulder_line[index_half].y) {
					shoulder_line_for_arm_test = possible_lines_for_arm[i];
					break;
				}
			}
		}

		if (!shoulder_line_for_arm_test.empty()) {
			for (int i = 0; i < shoulder_line_for_arm_test.size() - 1; i++) {
				line(shoulder_detection_image, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], yellow, 5, 8, 0);

				//THESIS
				//line(userInputFrame, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], green, 5, 8, 0);
			}

			//THESIS
			//circle(userInputFrame, shoulder_line_for_arm_test[0], 7, green, -1, 8);
		}

		//Check if the longest arm line is overlap some points of shoulder lines
		Refine_Overlap_ShoulderLine_And_ArmLine(shoulder_detection_image, shoulder_line, shoulder_line_for_arm_longest);

		//Add more point for fail detections
		Improve_Fail_Detected_ShoulderLine(shoulder_detection_image, shoulder_line, shoulderModel.head_upper_shoulder, shoulderModel.angle);
		//THESIS
		//circle(userInputFrame, shoulder_line.back(), 7, green, -1, 8);
		return shoulder_line;
	}
}

vector<Point2f> MYcppGui::Finding_ShoulderLines_From_PointCollection_v2(Mat shoulder_detection_image, vector<vector<Point2f>> point_collection, ShoulderModel &shoulderModel, Scalar color, double epsilon_angle_shoulder) {

	int angle_for_arm = RIGHT_ARM_ANGLE;
	if (shoulderModel.leftHandSide)
		angle_for_arm = LEFT_ARM_ANGLE;

	//Show possible point
	for (int a = 0; a < point_collection.size() - 1; a++) {
		for (int b1 = 0; b1 < point_collection[a].size(); b1++) {
			for (int b2 = 0; b2 < point_collection[a + 1].size(); b2++) {
				//Check difference of angle
				if (AngleDifference(Angle(point_collection[a][b1], point_collection[a + 1][b2]), shoulderModel.angle) <= epsilon_angle_shoulder
					|| AngleDifference(Angle(point_collection[a][b1], point_collection[a + 1][b2]), angle_for_arm) <= 30) {
					if (EuclideanDistance(point_collection[a][b1], point_collection[a + 1][b2]) < checking_block * 1.5) {
						line(shoulder_detection_image, point_collection[a][b1], point_collection[a + 1][b2], red, 3, 8, 0);

						//THESIS
						//line(userInputFrame, point_collection[a][b1], point_collection[a + 1][b2], red, 2, 8, 0);
					}

				}
			}
		}
	}

	vector<vector<PointPostion>> possible_lines_map;
	vector<vector<PointPostion>> possible_lines_for_arm_map;

	for (int i = 0; i < point_collection.size(); i++) {
		for (int j = 0; j < point_collection[i].size(); j++) {
			vector<PointPostion> path_map;
			vector<PointPostion> path_for_arm_map;

			if (i <= 13) {
				path_map = findPath_v2(i, j, point_collection, shoulderModel.angle, epsilon_angle_shoulder, 0, shoulderModel);
			}
			
			if (i >= 5) {
				path_for_arm_map = findPath_v2(i, j, point_collection, angle_for_arm, 30, 1, shoulderModel);
			}
			
			if (possible_lines_map.empty() || path_map.size() > possible_lines_map.back().size()) {
				if (!path_map.empty()) {
					possible_lines_map.push_back(path_map);
				}
			}

			if (possible_lines_for_arm_map.empty() || path_for_arm_map.size() > possible_lines_for_arm_map.back().size()) {
				if (!path_for_arm_map.empty()) {
					possible_lines_for_arm_map.push_back(path_for_arm_map);
				}
			}
		}
	}

	//old way use the longest one
	vector<Point2f> shoulder_line_for_arm_longest;
	vector<PointPostion> shoulder_line_for_arm_longest_map;
	if (!possible_lines_for_arm_map.empty()) {
		shoulder_line_for_arm_longest_map = possible_lines_for_arm_map.back();

		shoulder_line_for_arm_longest = ConvertFromMap(point_collection, shoulder_line_for_arm_longest_map);

		for (int i = 0; i < shoulder_line_for_arm_longest.size() - 1; i++) {
			line(shoulder_detection_image, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], black, 5, 8, 0);

			//THESIS
			//line(userInputFrame, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], black, 5, 8, 0);
		}
	}

	vector<PointPostion> shoulder_line_map;
	vector<Point2f> shoulder_line;

	if (!possible_lines_map.empty()) {
		shoulder_line_map = possible_lines_map.back();
		shoulder_line = ConvertFromMap(point_collection, shoulder_line_map);

		for (int i = 0; i < shoulder_line.size() - 1; i++) {
			line(shoulder_detection_image, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);

			// THESIS
			//line(userInputFrame, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);
		}


		//new way which is to detect the position of arm line
		//list use pushback, so the last one is [0]
		int index_one_third = shoulder_line.size() * 1 / 3;
		int index_half = shoulder_line.size() / 2;
		vector<Point2f> shoulder_line_for_arm_test;
		vector<PointPostion> shoulder_line_for_arm_test_map;

		for (int i = possible_lines_for_arm_map.size() - 1; i >= 0; i--) {
			if (abs(shoulder_line[0].x - possible_lines_for_arm_map[i].back().getFrom(point_collection).x) < abs(shoulder_line[0].x - shoulder_line[index_one_third].x))
			{
				if (possible_lines_for_arm_map[i].back().getFrom(point_collection).y > shoulder_line[index_half].y) {
					shoulder_line_for_arm_test_map = possible_lines_for_arm_map[i];
					break;
				}
			}
		}

		shoulder_line_for_arm_test = ConvertFromMap(point_collection, shoulder_line_for_arm_test_map);

		if (!shoulder_line_for_arm_test.empty()) {
			for (int i = 0; i < shoulder_line_for_arm_test.size() - 1; i++)
			{
				line(shoulder_detection_image, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], yellow, 5, 8, 0);

				//THESIS
				//line(userInputFrame, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], green, 5, 8, 0);
			}

			//THESIS
			//circle(userInputFrame, shoulder_line_for_arm_test[0], 7, green, -1, 8);
		}

		//Find armline from end nodes of shoulder line
		for (int i = 0; i < 3; i++) {
			vector<Point2f> path_for_arm = findPath(shoulder_line_map[0].index_line, shoulder_line_map[0].index, point_collection, angle_for_arm, 30, 1);
			if (path_for_arm.size() > 2){
				for (int j = 0; j < path_for_arm.size() - 1; j++) {
					int indx_color = (i + 3) % 8;
					line(shoulder_detection_image, path_for_arm[j], path_for_arm[j + 1], colors[indx_color], 5, 8, 0);
				}
			}
		}

		//Check if the longest arm line is overlap some points of shoulder lines
		Refine_Overlap_ShoulderLine_And_ArmLine(shoulder_detection_image, shoulder_line, shoulder_line_for_arm_longest);

		int sign = 1;		//GO RIGHT
		if (shoulderModel.leftHandSide) {	//GO LEFT
			sign = -1;
		}

		if (shoulder_line.size() <= 6 || (shoulder_line.back().x - shoulderModel.head_bottom_shoulder.x) *(shoulder_line.back().x - shoulderModel.end_bottom_shoulder.x + sign*checking_block) > 0) {

			Point2f pA = Point2f(max(left_cheek.x - shoulderModel.range_of_shoulder_sample*1.6, 0.0), min(left_cheek.y, right_cheek.y));
			Point2f pB = Point2f(min(right_cheek.x + shoulderModel.range_of_shoulder_sample*1.6, double(originalFrame.cols)), pA.y);
			Point2f pC = Point2f(pB.x, min(symmetric_point.y, float(originalFrame.rows)));
			Point2f pD = Point2f(pA.x, pC.y);
			Mat detected_edges;
			Mat Bigger_detected_edges(originalFrame.rows, originalFrame.cols, CV_8UC1, Scalar(0));
			Mat sub_frame = originalFrame(cv::Rect(pA.x, pA.y, pB.x - pA.x, pD.y - pA.y));

			detected_edges = StrongPreprocessing(sub_frame);
			detected_edges.copyTo(Bigger_detected_edges(cv::Rect(pA.x, pA.y, detected_edges.cols, detected_edges.rows)));
			shoulder_line = detectShoulderLine(shoulder_detection_image, Bigger_detected_edges, shoulderModel.leftHandSide, shoulderModel.angle, color, false, false);
		}

		//Add more point for fail detections
		Improve_Fail_Detected_ShoulderLine(shoulder_detection_image, shoulder_line, shoulderModel.head_upper_shoulder, shoulderModel.angle);
		//THESIS
		//circle(userInputFrame, shoulder_line.back(), 7, green, -1, 8);
		
	}
	return shoulder_line;
}

void MYcppGui::RefinePoint_collection(Mat& frame, vector<vector<Point2f>> &point_collection) {
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

//Inside too complicated
void MYcppGui::RefinePoint_collection_v21(Mat& frame, vector<vector<Point2f>> &point_collection) {
	for (int j = 0; j < point_collection.size(); j++) {
		vector<Point2f> point_line = point_collection[j];
		vector<Point2f> refine_points;

		for (int i = 0; i < (int)point_line.size() - 1; i++) {
			if (EuclideanDistance(point_line[i], point_line[i + 1]) > checking_block*0.8) {
				circle(frame, point_line[i], 5, blue, -1, 8);
				circle(frame, point_line[i + 1], 5, blue, -1, 8);
				refine_points.push_back(point_line[i]);
				refine_points.push_back(point_line[i + 1]);
				i++;
			}
		}

		if (point_line.size() != 0){
			//Add the last point
			if (refine_points.empty() || refine_points.back() != point_line.back()){
				refine_points.push_back(point_line.back());
				circle(frame, point_line.back(), 5, blue, -1, 8);
			}
			point_collection[j] = refine_points;
		}
	}
}

//Outside too complicated		//Still need improvement
void MYcppGui::RefinePoint_collection_v22(Mat& frame, vector<vector<Point2f>> &point_collection, ShoulderModel &shoulderModel) {
	for (int j = 0; j < point_collection.size(); j++) {
		vector<Point2f> point_line = point_collection[j];
		vector<Point2f> refine_points;

		double maxColourDistance = 0;
		int index01 = -2, index02 = -1;
		bool checkDistanceColor = false;

		for (int i = (int)point_line.size() - 2; i >= 0; i--) {
			//Add the first point which is only far from upper shoulder
			if (i + 1 == point_line.size() - 1) {
				if (DistanceFromPointToLine(point_line[i+1], shoulderModel.head_upper_shoulder, shoulderModel.end_upper_shoulder) > 1 * checking_block) {
						refine_points.push_back(point_line.back());
						circle(frame, point_line.back(), 5, blue, -1, 8);
						checkDistanceColor = true;
				}
			}



			if (EuclideanDistance(point_line[i], point_line[i + 1]) > checking_block * 0.8) {
				circle(frame, point_line[i], 5, blue, -1, 8);
				circle(frame, point_line[i + 1], 5, blue, -1, 8);
				refine_points.push_back(point_line[i]);
				refine_points.push_back(point_line[i + 1]);
				//i--;
				
				//Add 2 points with max color distance
				if (checkDistanceColor && index01 >= 0 && index02 >=0) {
					if (!IsBelongToCollection(point_line[index01], refine_points)) {
						refine_points.push_back(point_line[index01]);
						circle(frame, point_line[index01], 3, yellow, -1, 8);
					}
					if (!IsBelongToCollection(point_line[index02], refine_points)) {
						refine_points.push_back(point_line[index02]);
						circle(frame, point_line[index02], 3, yellow, -1, 8);
					}
					circle(frame, point_line[index01], 1, yellow, -1, 8);
					circle(frame, point_line[index02], 1, yellow, -1, 8);

					maxColourDistance = 0;
					index01 = -2, index02 = -1;
				}

				checkDistanceColor = true;
			}
			else {
				//Get distance color
				if (checkDistanceColor) {
					//Color distance from point_line[i], point_line[i + 1]
					Vec3b color01 = AverageColor(originalFrame, point_line[i].x, point_line[i].y);
					Vec3b color02 = AverageColor(originalFrame, point_line[i + 1].x, point_line[i + 1].y);
					double colourDistance = ColourDistance(color01, color02);

					//Save the max color distance
					if (colourDistance > maxColourDistance) {
						maxColourDistance = colourDistance;
						index01 = i;
						index02 = i + 1;
					}
				}
			}

			//Add the last point
			//Avoid redundant adding point
			if (i == 0 && (refine_points.empty() || refine_points.back() != point_line[0])){
				refine_points.push_back(point_line[0]);
				circle(frame, point_line[0], 5, blue, -1, 8);
				if (checkDistanceColor && index01 >= 0 && index02 >= 0) {
					if (!IsBelongToCollection(point_line[index01], refine_points)) {
						refine_points.push_back(point_line[index01]);
						circle(frame, point_line[index01], 3, yellow, -1, 8);
					}
					if (!IsBelongToCollection(point_line[index02], refine_points)) {
						refine_points.push_back(point_line[index02]);
						circle(frame, point_line[index02], 3, yellow, -1, 8);
					}
					circle(frame, point_line[index01], 1, yellow, -1, 8);
					circle(frame, point_line[index02], 1, yellow, -1, 8);
				}
			}
		}

		//The case point_line has 1 point
		if (point_line.size() == 1) {
			refine_points.push_back(point_line.back());
		}

		if (point_line.size() != 0){			
			point_collection[j] = refine_points;
		}
	}
}

void MYcppGui::RefinePoint_collection_v23(Mat& frame, vector<vector<Point2f>> &point_collection, ShoulderModel &shoulderModel) {
	//A -- B
	//|    | 
	//D -- C
	Point2f pA = Point2f(max(left_cheek.x - shoulderModel.range_of_shoulder_sample*1.6, 0.0), min(left_cheek.y, right_cheek.y));
	Point2f pB = Point2f(min(right_cheek.x + shoulderModel.range_of_shoulder_sample*1.6, double(originalFrame.cols)), pA.y);
	Point2f pC = Point2f(pB.x, min(symmetric_point.y, float(originalFrame.rows)));
	Point2f pD = Point2f(pA.x, pC.y);
	Mat detected_edges;
	Mat Bigger_detected_edges(originalFrame.rows, originalFrame.cols, CV_8UC1, Scalar(0));
	Mat sub_frame = originalFrame(cv::Rect(pA.x, pA.y, pB.x - pA.x, pD.y - pA.y));

	detected_edges = StrongPreprocessing(sub_frame);
	detected_edges.copyTo(Bigger_detected_edges(cv::Rect(pA.x, pA.y, detected_edges.cols, detected_edges.rows)));

	vector<vector<Point2f>> new_point_collection = Collect_Potential_ShoulderPoint(frame, Bigger_detected_edges, shoulderModel, false, false, green);
	
	point_collection = new_point_collection;
}

int MYcppGui::RefinePoint_collection_v2(Mat& shoulder_detection_image, Mat &detected_edges, vector<vector<Point2f>> &point_collection, ShoulderModel &shoulderModel){
	double height = 30;
	int type = 0;
	
	Point2f intersection_point;
	intersection(shoulderModel.head_upper_shoulder, symmetric_point, shoulderModel.head_bottom_shoulder, shoulderModel.end_bottom_shoulder, intersection_point);

	Point2f bottom_shoulder_point(intersection_point.x, min(intersection_point.y + height, (double)shoulder_detection_image.size().height));

	double Percent_BG = Check_Density(shoulder_detection_image, detected_edges, shoulderModel.head_upper_shoulder, shoulderModel.angle, height);
	double Percent_FG = Check_Density(shoulder_detection_image, detected_edges, bottom_shoulder_point, shoulderModel.angle, height);

	cout << endl << "	Percent_BG: " << Percent_BG << endl;
	cout << "	Percent_FG: " << Percent_FG << endl;

	if (Percent_BG <= 0.3 && Percent_FG >= 0.5) {		//Inside too complicated
		RefinePoint_collection_v21(shoulder_detection_image, point_collection);
		type = 1;
	}
	else if (Percent_BG <= 0.2 && Percent_FG >= 0.4) {		//Inside too complicated
		RefinePoint_collection_v21(shoulder_detection_image, point_collection);
		type = 1;
	}


	if (Percent_BG >= 0.5 && Percent_FG <= 0.3) {		//Outside too complicated
		RefinePoint_collection_v22(shoulder_detection_image, point_collection, shoulderModel);
		type = 1;
	}
	else if (Percent_BG >= 0.4 && Percent_FG <= 0.2) {		//Outside too complicated
		RefinePoint_collection_v22(shoulder_detection_image, point_collection, shoulderModel);
		type = 1;
	}

	if (Percent_BG >= 0.4 && Percent_FG >= 0.4) {
		RefinePoint_collection_v23(shoulder_detection_image, point_collection, shoulderModel);
	}

	return type;
}

double MYcppGui::Check_Density(Mat& shoulder_detection_image, Mat &detected_edges, Point2f head_bottom_point, int angle, double height) {
	double radian = angle * CV_PI / 180.0;
	//upper shoulder sample line

	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x); //used to be *2
	double length = abs(range_of_shoulder_sample / cos(radian));

	double All = 0;
	double Edge = 0;
	for (int h = 0; h <= height; h += 10) {
		Point2f start_point = Point2f(head_bottom_point.x, head_bottom_point.y - h);
		
		for (int j = 0; abs(checking_block/4*j*cos(radian)) < range_of_shoulder_sample; j++) {
			Point2f current_point = Point2f(start_point.x + checking_block / 4 * j*cos(radian), start_point.y - checking_block / 4 * j*sin(radian));
			int value_in_edge_map = 0;
			
			// Skip the point out of the frame
			if (current_point.x < 0 || current_point.x > shoulder_detection_image.size().width - 1 || current_point.y > shoulder_detection_image.size().height - 1) {
				continue;
			}

			value_in_edge_map = detected_edges.at<uchar>(current_point);
			if (value_in_edge_map == 255) {
				Edge++;
				circle(shoulder_detection_image, current_point, 7, white, -1, 8);
			}
			All++;
		}
	}
	double percentage = Edge / All;
	return percentage;
}

vector<Point2f> MYcppGui::Finding_ShoulderLines_From_PointCollection_v3(Mat shoulder_detection_image, vector<vector<Point2f>> point_collection, bool leftHandSide, int angle, Scalar color) {

	double radian = angle * CV_PI / 180.0;

	//upper shoulder sample line
	Point2f head_upper_shoulder;

	if (leftHandSide) {	//GO LEFT
		head_upper_shoulder = Point2f(left_cheek.x - 10, left_cheek.y);	// - distance_from_face_to_shouldersample
	}
	else {
		head_upper_shoulder = Point2f(right_cheek.x + 10, right_cheek.y);	// + distance_from_face_to_shouldersample
	}

	int angle_for_arm = RIGHT_ARM_ANGLE;
	if (leftHandSide)
		angle_for_arm = -LEFT_ARM_ANGLE;

	//check ki trc khi xoa
	for (int a = 0; a < point_collection.size() - 1; a++) {
		for (int b1 = 0; b1 < point_collection[a].size(); b1++)
		{
			for (int b2 = 0; b2 < point_collection[a + 1].size(); b2++)
			{
				//Check difference of angle
				if (AngleDifference(Angle(point_collection[a][b1], point_collection[a + 1][b2]), angle) <= 40 
					|| AngleDifference(Angle(point_collection[a][b1], point_collection[a + 1][b2]), angle_for_arm) <= 20)
				{
					line(shoulder_detection_image, point_collection[a][b1], point_collection[a + 1][b2], red, 3, 8, 0);

					//THESIS
					//line(userInputFrame, point_collection[a][b1], point_collection[a + 1][b2], blue, 2, 8, 0);
				}
			}
		}
	}

	vector<vector<PointPostion>> possible_lines_map;
	vector<vector<PointPostion>> possible_lines_for_arm_map;

	int max_possible_lines_map = 0;
	int max_possible_lines_for_arm_map = 0;

	for (int i = 0; i < point_collection.size(); i++) {
		for (int j = 0; j < point_collection[i].size(); j++) {
			vector<vector<PointPostion>> path_map = findPath_new(i, j, point_collection, angle);			//Limit it later - loop
			vector<vector<PointPostion>> path_for_arm_map = findPath_new(i, j, point_collection, angle_for_arm);
			
			int max_path = 0;
			for (int i = 0; i < path_map.size(); i++) {						//Maybe dont need it
				max_path = max(max_path, (int)path_map[i].size());
			}

			if (possible_lines_map.empty() || max_path >= max_possible_lines_map)
			{
				if (!path_map.empty())
				{
					possible_lines_map.insert(possible_lines_map.end(), path_map.begin(), path_map.end());

					if (max_path > max_possible_lines_map) {
						max_possible_lines_map = max_path;
					}
				}
			}

			int max_path_for_arm = 0;
			for (int i = 0; i < path_for_arm_map.size(); i++) {						//Maybe dont need it
				max_path_for_arm = max(max_path_for_arm, (int)path_for_arm_map[i].size());
			}

			if (possible_lines_for_arm_map.empty() || max_path_for_arm >= max_possible_lines_for_arm_map)
			{
				if (!path_for_arm_map.empty())
				{
					possible_lines_for_arm_map.insert(possible_lines_for_arm_map.end(), path_for_arm_map.begin(), path_for_arm_map.end());

					if (max_path_for_arm > max_possible_lines_for_arm_map) {
						max_possible_lines_for_arm_map = max_path_for_arm;
					}
				}
			}
		}
	}

	//old way use the longest one
	vector<Point2f> shoulder_line_for_arm_longest;
	vector<PointPostion> shoulder_line_for_arm_longest_map;

	if (!possible_lines_for_arm_map.empty()) {
		shoulder_line_for_arm_longest_map = possible_lines_for_arm_map.back();

		shoulder_line_for_arm_longest = ConvertFromMap(point_collection, shoulder_line_for_arm_longest_map);
		for (int i = 0; i < shoulder_line_for_arm_longest.size() - 1; i++) {
			//line(shoulder_detection_image, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], black, 5, 8, 0);

			//THESIS
			//line(userInputFrame, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], green, 5, 8, 0);
		}
	}

	int no_color = 0;
	for (int k = possible_lines_map.size() - 2; k >= 0; k--) {
		vector<Point2f> shoulder_line;
		shoulder_line = ConvertFromMap(point_collection, possible_lines_map[k]);

		for (int i = 0; i < shoulder_line.size() - 1; i++) {
			line(shoulder_detection_image, shoulder_line[i], shoulder_line[i + 1], colors[no_color], 5, 8, 0);

			// THESIS
			//line(userInputFrame, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);
		}
		no_color++;
		if (no_color == 1 || no_color == 2) {
			no_color += 2;
		}
		no_color %= 8;
	}

	if (!possible_lines_map.empty()) {
		
		vector<PointPostion> shoulder_line_map = possible_lines_map.back();
		vector<Point2f> shoulder_line;

		//Convert shoulder_line_map into shoulder_line
		shoulder_line = ConvertFromMap(point_collection, shoulder_line_map);

		for (int i = 0; i < shoulder_line_map.size() - 1; i++) {
			line(shoulder_detection_image, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);
			
			// THESIS
			line(userInputFrame, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);
		}


		//new way which is to detect the position of arm line
		//list use pushback, so the last one is [0]
		int index_one_third = shoulder_line_map.size() * 1 / 3;
		int index_half = shoulder_line_map.size() / 2;
		vector<Point2f> shoulder_line_for_arm_test;
		vector<PointPostion> shoulder_line_for_arm_test_map;

		for (int i = possible_lines_for_arm_map.size() - 1; i >= 0; i--) {
			if (abs(shoulder_line[0].x - possible_lines_for_arm_map[i].back().getFrom(point_collection).x) 
				< abs(shoulder_line[0].x - shoulder_line[index_one_third].x)) {
				if (possible_lines_for_arm_map[i].back().getFrom(point_collection).y > shoulder_line[index_half].y) {
					shoulder_line_for_arm_test_map = possible_lines_for_arm_map[i];
					break;
				}
			}
		}

		if (!shoulder_line_for_arm_test_map.empty()) {
			shoulder_line_for_arm_test = ConvertFromMap(point_collection, shoulder_line_for_arm_test_map);

			for (int i = 0; i < shoulder_line_for_arm_test.size() - 1; i++) {
				line(shoulder_detection_image, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], yellow, 5, 8, 0);

				//THESIS
				//line(userInputFrame, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], green, 5, 8, 0);
			}

			//THESIS
			//circle(userInputFrame, shoulder_line_for_arm_test[0], 7, green, -1, 8);
		}

		//Check if the longest arm line is overlap some points of shoulder lines
		Refine_Overlap_ShoulderLine_And_ArmLine(shoulder_detection_image, shoulder_line, shoulder_line_for_arm_longest);

		//Add more point for fail detections
		Improve_Fail_Detected_ShoulderLine(shoulder_detection_image, shoulder_line, head_upper_shoulder, angle);
		//THESIS
		//circle(userInputFrame, shoulder_line.back(), 7, green, -1, 8);
		return shoulder_line;
	}
}

//Check if the longest arm line is overlap some points of shoulder lines
void MYcppGui::Refine_Overlap_ShoulderLine_And_ArmLine(Mat &shoulder_detection_image, vector<Point2f> &shoulder_line, vector<Point2f> &shoulder_line_for_arm_longest) {
	// Construct sub_shoulder_line is equal to first half shoulder ==> make sure the step below is more correct
	vector<Point2f> sub_shoulder_line(shoulder_line.begin(), shoulder_line.begin() + shoulder_line.size() / 2);
	vector<Point2f> sub_arm_line(shoulder_line_for_arm_longest.begin() + shoulder_line_for_arm_longest.size() / 2, shoulder_line_for_arm_longest.end());

	if (CheckCommon(sub_shoulder_line, shoulder_line_for_arm_longest)) {
		for (int i = 0; i < shoulder_line_for_arm_longest.size(); i++) {
			for (int j = 0; j < shoulder_line.size(); j++) {
				//First common points
				if (shoulder_line[j] == shoulder_line_for_arm_longest[i]) {
					if (j + 1 < shoulder_line.size() && i + 1 < shoulder_line_for_arm_longest.size()) {
						//Check if there is a second point
						if (shoulder_line[j + 1] == shoulder_line_for_arm_longest[i + 1]) {

							// Remove the first N elements, and shift everything else down by N indices
							for (int k = 0; k < j; k++){
								line(shoulder_detection_image, shoulder_line[k], shoulder_line[k + 1], black, 5, 8, 0);
								
								//Thesis
								line(userInputFrame, shoulder_line[k], shoulder_line[k + 1], black, 5, 8, 0);
							}
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
								line(shoulder_detection_image, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], green, 5, 8, 0);
								
								//Thesis
								line(userInputFrame, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], green, 5, 8, 0);
							}
							return;

						}
					}
				}
			}
		}
	}

}

//Add more point for fail detections
void MYcppGui::Improve_Fail_Detected_ShoulderLine(Mat &shoulder_detection_image, vector<Point2f> &shoulder_line, Point2f head_upper_shoulder, int angle) {
	double radian = angle * CV_PI / 180.0;

	bool is_intersect = false;
	int j = 2;
	Point2f upper_shoulder_point02 = Point2f(head_upper_shoulder.x + checking_block*j*cos(radian),
		head_upper_shoulder.y - checking_block*j*sin(radian));
	Point2f above_chin = Point2f(chin.x, chin.y - checking_block);

	if (doIntersect(upper_shoulder_point02, symmetric_point, above_chin, shoulder_line.back())) {
		Point2f intersection_point;
		intersection(upper_shoulder_point02, symmetric_point, above_chin, shoulder_line.back(), intersection_point);
		line(shoulder_detection_image, intersection_point, shoulder_line.back(), blue, 5, 8, 0);

		//THESIS
		//line(userInputFrame, intersection_point, shoulder_line.back(), green, 5, 8, 0);
		shoulder_line.push_back(intersection_point);
	}

	j = 1;
	Point2f upper_shoulder_point01 = Point2f(head_upper_shoulder.x + checking_block*j*cos(radian),
		head_upper_shoulder.y - checking_block*j*sin(radian));
	Point2f above_chin02 = Point2f(chin.x, chin.y - checking_block*2);
	circle(shoulder_detection_image, above_chin, 5, green, -1, 8);

	if (doIntersect(upper_shoulder_point01, symmetric_point, above_chin02, shoulder_line.back())) {
		Point2f intersection_point;
		intersection(upper_shoulder_point01, symmetric_point, above_chin02, shoulder_line.back(), intersection_point);
		line(shoulder_detection_image, intersection_point, shoulder_line.back(), blue, 5, 8, 0);

		//THESIS
		//line(userInputFrame, intersection_point, shoulder_line.back(), green, 5, 8, 0);

		shoulder_line.push_back(intersection_point);
	}
}

//type 0: shoulder - 1: arm - 2:neck
vector<Point2f> MYcppGui::findPath(int index_line, int index, vector<vector<Point2f>> &point_collection, double angle, double epsilon, int type) {
	if (index_line >= point_collection.size() - 1) {
		vector<Point2f> result;
		result.push_back(point_collection[index_line][index]);
		return result;
	}

	vector<Point2f> new_point_line;
	vector<Point2f> tmp_new_point_line;

	//control angle at  the middle of shoulder line
	if (index_line == 10 && type == 0) {
		if (angle < -90)
			angle += 5;
		else
			angle -= 5;
		epsilon -= 5;
	}

	if (index_line == 13 && type == 0) {
		if (angle < -90)
			angle += 5;
		else
			angle -= 5;
	}

	//In case that next index_line have no point ==> move until 
	int k = 1;
	if (point_collection[index_line + 1].size() == 0 && index_line + 2 < point_collection.size()) {
		k = 2;
	}

	for (int i = point_collection[index_line + k].size() - 1; i >= 0; i--) {	 	//Go inside out to choose the outter result
		Point2f current_point = point_collection[index_line][index];
		Point2f next_point = point_collection[index_line + k][i];
		//check angle
		if (AngleDifference(Angle(current_point, next_point), angle) <= epsilon) { //used to 25
			if (EuclideanDistance(current_point, next_point) < checking_block * 1.5) {		//new condition
				tmp_new_point_line = findPath(index_line + k, i, point_collection, angle, epsilon, type);
			}

		}
		tmp_new_point_line.push_back(point_collection[index_line][index]);

		if (new_point_line.empty() || tmp_new_point_line.size() > new_point_line.size())
		{
			new_point_line = tmp_new_point_line;
		}
		tmp_new_point_line.clear();
	}

	//The case that last chosen point was missing because the next index_line 's size  == 0
	if (point_collection[index_line + k].size() == 0) {
		tmp_new_point_line.push_back(point_collection[index_line][index]);

		if (new_point_line.empty() || tmp_new_point_line.size() > new_point_line.size())
		{
			new_point_line = tmp_new_point_line;
		}
		tmp_new_point_line.clear();
	}
	return new_point_line;
}

//type: 0: Shoulder  -  1: Arm  -  2: Neck 
vector<PointPostion> MYcppGui::findPath_v2(int index_line, int index, vector<vector<Point2f>> &point_collection, double angle, double epsilon, int type, ShoulderModel &shoulderModel) {

	if (index_line >= point_collection.size() - 1) {
		vector<PointPostion> result;
		result.push_back(PointPostion(index_line, index));
		return result;
	}

	vector<PointPostion> new_point_line;
	vector<PointPostion> tmp_new_point_line;

	//control angle at  the middle of shoulder line
	if (index_line == 10 && type == 0) {
		if (angle < -90)
			angle += 5;
		else
			angle -= 5;
		epsilon -= 5;
	}

	if (index_line == 13 && type == 0) {
		if (angle < -90)
			angle += 5;
		else
			angle -= 5;
	}

	//In case that next index_line have no point ==> move until 
	int k = 1;
	if (point_collection[index_line + 1].size() == 0 && index_line + 2 < point_collection.size()) {
		k = 2;
	}

	for (int i = point_collection[index_line + k].size() - 1; i >= 0; i--)		//Go inside out to choose the outter result
		// I will take all later and pick the closest one to previous result
	{
		Point2f current_point = point_collection[index_line][index];

		//Do not take the point inside 1 range.
		if (VIDEO_MODE) {
			if (current_point.y < chin.y) {
				continue;
			}
			if (type == 1) {
				if ((current_point.x - chin.x)*(current_point.x - shoulderModel.end_bottom_shoulder.x) < 0) {
					continue;
				}
			}
		}

		Point2f next_point = point_collection[index_line + k][i];
		//check angle
		if (AngleDifference(Angle(current_point, next_point), angle) <= epsilon) {			//used to 25
			if (EuclideanDistance(current_point, next_point) < checking_block * 1.5) {		//new condition
				tmp_new_point_line = findPath_v2(index_line + k, i, point_collection, angle, epsilon, type, shoulderModel);
			}

		}
		tmp_new_point_line.push_back(PointPostion(index_line, index));

		if (new_point_line.empty() || tmp_new_point_line.size() > new_point_line.size())
		{
			new_point_line = tmp_new_point_line;
		}
		tmp_new_point_line.clear();
	}

	//The case that last chosen point was missing because the next index_line 's size  == 0
	if (point_collection[index_line + k].size() == 0) {
		tmp_new_point_line.push_back(PointPostion(index_line, index));

		if (new_point_line.empty() || tmp_new_point_line.size() > new_point_line.size())
		{
			new_point_line = tmp_new_point_line;
		}
		tmp_new_point_line.clear();
	}

	return new_point_line;
}

vector<vector<PointPostion>> MYcppGui::findPath_new(int index_line, int index, vector<vector<Point2f>> &point_collection, double angle) {
	if (index_line >= point_collection.size() - 1) {
		vector<vector<PointPostion>> result;
		for (int i = 0; i < result.size(); i++) {
			result[i].push_back(PointPostion(index_line, index));
		}

		return result;
	}

	vector<vector<PointPostion>> new_point_lines;
	vector<vector<PointPostion>> tmp_new_point_lines;

	//In case that next index_line have no point ==> move until 
	int k = 1;
	if (point_collection[index_line + 1].size() == 0 && index_line + 2 < point_collection.size()) {
		k = 2;
	}

	int max_tmp_new_point_lines = 0;
	int max_new_point_lines = 0;

	for (int i = point_collection[index_line + k].size() - 1; i >= 0; i--) {		//Go inside out to choose the outter result	
		Point2f current_point = point_collection[index_line][index];
		Point2f next_point = point_collection[index_line + k][i];
		//check angle
		if (AngleDifference(Angle(current_point, next_point), angle) <= 30) { //used to 25
			if (EuclideanDistance(current_point, next_point) < checking_block * 2) {		//new condition
				tmp_new_point_lines = findPath_new(index_line + k, i, point_collection, angle);
			}
		}

		//Add for all lines from next index_line
		for (int j = 0; j < tmp_new_point_lines.size(); j++) {
			tmp_new_point_lines[j].push_back(PointPostion(index_line, index));
			//Update max of tmp_new_point_lines
			max_tmp_new_point_lines = max(max_tmp_new_point_lines, (int)tmp_new_point_lines[j].size());
		}

		if (new_point_lines.empty() || max_tmp_new_point_lines == max_new_point_lines) {	//Can change later
			//concatenate two vectors
			new_point_lines.insert(new_point_lines.end(), tmp_new_point_lines.begin(), tmp_new_point_lines.end());
			max_new_point_lines = max_tmp_new_point_lines;
		}
		else if (max_tmp_new_point_lines > max_new_point_lines) {				//Choose the longest lines
			new_point_lines = tmp_new_point_lines;
			max_new_point_lines = max_tmp_new_point_lines;
		}

		tmp_new_point_lines.clear();
	}

	if (point_collection[index_line + k].size() != 0 && new_point_lines.empty()) {
		new_point_lines.push_back(vector<PointPostion>{PointPostion(index_line, index)});
		max_new_point_lines = 1;
	}

	//The case that last chosen point was missing because the next index_line 's size  == 0
	if (point_collection[index_line + k].size() == 0) {

		//Add for all lines from next index_line
		for (int j = 0; j < tmp_new_point_lines.size(); j++) {
			tmp_new_point_lines[j].push_back(PointPostion(index_line, index));
			max_tmp_new_point_lines = max(max_tmp_new_point_lines, (int)tmp_new_point_lines[j].size());
		}
		if (tmp_new_point_lines.empty()) {
			tmp_new_point_lines.push_back(vector<PointPostion>{PointPostion(index_line, index)});
			max_tmp_new_point_lines = 1;
		}

		if (new_point_lines.empty() || max_tmp_new_point_lines == max_new_point_lines)		//Can change later
		{
			//concatenate two vectors
			new_point_lines.insert(new_point_lines.end(), tmp_new_point_lines.begin(), tmp_new_point_lines.end());
		}
		else if (max_tmp_new_point_lines > max_new_point_lines) {				//Choose the longest lines
			new_point_lines = tmp_new_point_lines;
			max_new_point_lines = max_tmp_new_point_lines;
		}

		tmp_new_point_lines.clear();
	}
	return new_point_lines;
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

Mat MYcppGui::StrongPreprocessing(Mat frame) {
	Mat detected_edges;
	Mat tmp;	//To fix the problem when do morpholofy operation with sub_frame;
	//--------------------------------blur ----------------------------
	int blurIndex = 7;
	medianBlur(frame, tmp, blurIndex);

	//--------------------------------Morphology Open Close ----------------------------
	Morphology_Operations(tmp);

	//----------------------------------Canny ---------------------
	CannyProcessing(tmp, detected_edges);

	//----------------------------- Erosion after canny --------------------
	int erosion_size = 6;

	Mat element = getStructuringElement(cv::MORPH_CROSS,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	cv::dilate(detected_edges, detected_edges, element);
	return detected_edges;
}

void MYcppGui::collectColorShoulderFromInput(Mat &frame) {
	for (int k = 0; k < 2; k++) {	
		for (int i = 0; i < userInput[k].size(); i+=5) {
			//Get color inside 25px point
			int x = userInput[k][i].x;
			int y = userInput[k][i].y + 18;
			Vec3b color = frame.at<Vec3b>(Point2f(x, y));
		
			if (i == 0 && k == 0) {
				colorCollection_Shoulder.push_back(color);
				continue;
			}

			double minColourDistance = ColourDistance(color, colorCollection_Shoulder[0]);
			//cout << minColourDistance << endl;

			for (int j = 1; j < colorCollection_Shoulder.size(); j++) {
				double colourDistance = ColourDistance(color, colorCollection_Shoulder[j]);
				if (colourDistance < minColourDistance)
					minColourDistance = colourDistance;
			}

			circle(frame, Point2f(x, y), 5, green, -1, 8);

			if (minColourDistance > 20) {
				colorCollection_Shoulder.push_back(color);
				circle(frame, Point2f(x, y), 5, blue, -1, 8);
			}
		}
	}
}

double ColourDistance(Vec3b e1, Vec3b e2) {
	//0 - blue 
	//1 - green
	//2 - red
	long rmean = ((long)e1[2] + (long)e2[2]) / 2;
	long r = (long)e1[2] - (long)e2[2];
	long g = (long)e1[1] - (long)e2[1];
	long b = (long)e1[0] - (long)e2[0];
	return sqrt((((512 + rmean)*r*r) >> 8) + 4 * g*g + (((767 - rmean)*b*b) >> 8));
}



bool MYcppGui::IsMatchToColorCollectionInput(Vec3b color) {

	for (int i = 0; i < colorCollection_Shoulder.size(); ++i) {
		if (ColourDistance(color, colorCollection_Shoulder[i]) <= 30){
			return true;
		}
	}
	return false;
}

bool MYcppGui::IsMatchColor(Vec3b color, Vector<Vec3b> Collection, int epsilon) {
	for (int i = 0; i < Collection.size(); ++i) {
		if (ColourDistance(color, Collection[i]) <= epsilon){
			return true;
		}
	}
	return false;
}

double EuclideanDistance(Point2f p1, Point2f p2) {
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)); 
}


double Angle(Point2f start, Point2f end) {
	const double Rad2Deg = 180.0 / CV_PI;
	const double Deg2Rad = CV_PI / 180.0;
	return atan2(start.y - end.y, end.x - start.x) * Rad2Deg;
}

double AngleDifference(double angleA, double angleB) {
	double difference;
	difference = abs(angleA - angleB);
	difference = difference > 180 ? 360 - difference : difference;
	return difference;
}


float FindY_LineEquationThroughTwoPoint(float x_, Point2f p1, Point2f p2) {
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

Point2f mirror(Point2f p, Point2f point0, Point2f point1) {
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
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
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
	if (!capture.isOpened())			// if not success, exit program
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

vector<Point2f> ConvertFromMap(vector<vector<Point2f>> point_collection, vector<PointPostion> map) {
	vector<Point2f> line;
	for (int i = 0; i < map.size(); i++) {
		line.push_back(map[i].getFrom(point_collection));
	}
	return line;
}

void getLine(Point2f &p1, Point2f &p2, double &a, double &b, double &c)
{
	// (x- p1X) / (p2X - p1X) = (y - p1Y) / (p2Y - p1Y) 
	a = p1.y - p2.y;
	b = p2.x - p1.x;
	c = p1.x * p2.y - p2.x * p1.y;
}

double DistanceFromPointToLine(Point2f &p, Point2f &p2, Point2f &p3) {
	double a, b, c;
	getLine(p2, p3, a, b, c);
	return abs(a * p.x + b * p.y + c) / sqrt(a * a + b * b);
}

bool IsBelongToCollection(Point2f point, vector<Point2f> point_line) {
	for (int i = 0; i < point_line.size(); i++) {
		if (point_line[i] == point) {
			return true;
		}
	}
	return false;
}