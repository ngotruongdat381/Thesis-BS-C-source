#pragma once

#include "cppOpencv.h"

//using namespace std;
//using namespace cv;

MYcppGui::MYcppGui()
{
	checking_block = 0;
	dlib::deserialize("D:\\shape_predictor_68_face_landmarks.dat") >> shape_predictor;
}

MYcppGui::~MYcppGui()
{
	cvDestroyAllWindows();
}
void MYcppGui::AddUserInput(vector<cv::Point> _userInput)
{
	userInput = _userInput;
	userInputFrame = NULL;
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

//Mat MYcppGui::face_detection_dlib(string fileName)
//{
//	cout << fileName << endl;
//	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
//
//	dlib::shape_predictor sp;
//	dlib::deserialize("D:\\shape_predictor_68_face_landmarks.dat") >> sp;
//
//	Mat frame;
//	Mat src;
//
//	//Load an image
//	frame = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
//
//	cv::resize(frame, src, cv::Size(), 0.75, 0.75);
//	//src = frame;
//
//	dlib::array2d<dlib::rgb_pixel> cimg;
//	assign_image(cimg, dlib::cv_image<dlib::bgr_pixel>(src));
//	pyramid_up(cimg);
//
//	// Now tell the face detector to give us a list of bounding boxes
//	// around all the faces in the image.
//	std::vector<dlib::rectangle> dets = detector(cimg);
//	cout << "Number of faces detected: " << dets.size() << endl;
//
//	// Now we will go ask the shape_predictor to tell us the pose of
//	// each face we detected.
//	std::vector<dlib::full_object_detection> shapes;
//	cout << src.cols << "," << src.rows << endl;
//	cout << frame.cols << "," << frame.rows << endl;
//	double fraction = 1;
//	for (unsigned long j = 0; j < dets.size(); ++j)
//	{
//		dlib::full_object_detection shape = sp(cimg, dets[j]);
//		cout << "number of parts: " << shape.num_parts() << endl;
//		shapes.push_back(shape);
//		cout << endl;
//		for (int i = 0; i < shape.num_parts(); i++)
//		{
//			cout << Point(shape.part(i).x(), shape.part(i).y()) << endl;
//			circle(frame, Point(shape.part(i).x() / fraction, shape.part(i).y() / fraction), 7, green, -1, 8);
//		}
//	}
//
//	return frame;
//}

void MYcppGui::VideoProcessing(string fileName) {
	VideoCapture capture(fileName);
	if (!capture.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return;
	}
	double fps = capture.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;

	cv::namedWindow("Source", CV_WINDOW_NORMAL);
	cv::resizeWindow("Source", 282, 502);

	int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), true);

	
	while (1)
	{
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
		ImageProcessing_WithUserInput(face_processed, false);
		cv::imshow("Source", face_processed);
		video.write(face_processed);
		nth++;

		if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
		
	}
}

//!Process with a image
//void MYcppGui::ImageProcessing(Mat &frame) {
//	int blurIndex = 7;
//
//	Mat src, face_detection_frame; // delete soon
//
//	//backup the frame 
//	src = frame.clone();
//
//	face_detection_frame = frame.clone(); // delete soon
//
//	cv::namedWindow("Source", CV_WINDOW_NORMAL);
//	cv::resizeWindow("Source", 282, 502);
//	cv::imshow("Source", src);
//
//
//	//--------------------------------blur - start----------------------------
//	medianBlur(frame, frame, blurIndex);
//
//	cv::namedWindow("Median Blur", CV_WINDOW_NORMAL);
//	cv::resizeWindow("Median Blur", 282, 502);
//	cv::imshow("Median Blur", frame);
//
//
//	//--------------------------------Morphology Open Close - start----------------------------
//	Morphology_Operations(frame); //return nothing
//
//	cv::namedWindow("Morphology Open Close", CV_WINDOW_KEEPRATIO);
//	cv::resizeWindow("Morphology Open Close", 282, 502);
//	cv::imshow("Morphology Open Close", frame);
//
//	//----------------------------------Canny - start ---------------------
//	Mat detected_edges;
//
//	CannyProcessing(frame, detected_edges);
//	cv::namedWindow("Edge Map", CV_WINDOW_KEEPRATIO);
//	cv::resizeWindow("Edge Map", 282, 502);
//	cv::imshow("Edge Map", detected_edges);
//
//
//	//----------------------------- Erosion after canny - start --------------------
//	// erosion_size
//	int erosion_size = 6;
//
//	Mat element = getStructuringElement(cv::MORPH_CROSS,
//										Size(2 * erosion_size + 1, 2 * erosion_size + 1),
//										Point(erosion_size, erosion_size));
//	cv::dilate(detected_edges, detected_edges, element);
//
//	cv::namedWindow("Erosion After Canny", CV_WINDOW_KEEPRATIO);
//	cv::resizeWindow("Erosion After Canny", 282, 502);
//	cv::imshow("Erosion After Canny", detected_edges);
//
//
//	//----------------------------- Detect face - start ---------------------
//	double fraction = 1;
//	std::vector<dlib::full_object_detection> shapes_face;
//
//	shapes_face = face_detection_dlib_image(face_detection_frame);
//	detectNecessaryPointsOfFace(shapes_face);
//
//	circle(face_detection_frame, left_cheek, 7, green, -1, 8);
//	circle(face_detection_frame, right_cheek, 7, green, -1, 8);
//	circle(face_detection_frame, top_nose, 7, green, -1, 8);
//	circle(face_detection_frame, chin, 7, green, -1, 8);
//	circle(face_detection_frame, symmetric_point, 7, green, -1, 8);
//
//	double left_eye_width = shapes_face[0].part(39).x() - shapes_face[0].part(36).x();
//	double distance_from_face_to_shouldersample = left_eye_width * 3 / 4;
//	double checking_block = left_eye_width * 1 / 3;
//
//
//	//-----------------------------left shoulder---------------------------
//	double angle_left = -150;
//	double radian_left = angle_left * CV_PI / 180.0;
//	int length = 0; //500 before
//
//	length = abs((float)(symmetric_point.y - left_cheek.y) / sin(radian_left));
//
//	Point head_left_shoulder = Point(left_cheek.x - distance_from_face_to_shouldersample, left_cheek.y);
//	Point end_left_shoulder = Point(head_left_shoulder.x + length*cos(radian_left), head_left_shoulder.y - length*sin(radian_left));
//
//	detectShoulderLine(face_detection_frame, detected_edges, head_left_shoulder, end_left_shoulder, angle_left, checking_block);
//
//
//	//-----------------------------right shoulder---------------------------
//	int angle_right = -30;
//	double radian_right = angle_right * CV_PI / 180.0;
//
//	Point head_right_shoulder = Point(right_cheek.x + distance_from_face_to_shouldersample, right_cheek.y);
//	Point end_right_shoulder = Point(head_right_shoulder.x + length*cos(radian_right), head_right_shoulder.y - length*sin(radian_right));
//
//	detectShoulderLine(face_detection_frame, detected_edges, head_right_shoulder, end_right_shoulder, angle_right, checking_block);
//
//	cv::imshow("Erosion After Canny", detected_edges);
//	cv::imshow("Source", face_detection_frame);
//	
//	//return face_detection_frame
//	frame = Mat(face_detection_frame);
//
//}


Mat MYcppGui::ImageProcessing(string fileName, vector<cv::Point> userInput)
{
	cv::Mat img_input = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
	ImageProcessing_WithUserInput(img_input, true);
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
		
		//cout << "number of parts: " << shape.num_parts() << endl;
		//for (int i = 0; i < shape.num_parts(); i++) {
		//cout << "<part name='" << i << "' x='" << shape.part(i).x() << "' y='" << shape.part(i).y() << "'/>" << endl;
		//}

		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.

		new_shapes.push_back(shape);
	}
	std::cout << " Time for landmark face: " << float(clock() - tmp) / CLOCKS_PER_SEC << endl;

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

//!Detect neccessary points of face for next processing
void MYcppGui::detectNecessaryPointsOfFace(std::vector<dlib::full_object_detection> shapes_face) {
	double fraction = 1;

	left_cheek = Point(shapes_face[0].part(4).x() / fraction, shapes_face[0].part(4).y() / fraction);
	right_cheek = Point(shapes_face[0].part(12).x() / fraction, shapes_face[0].part(12).y() / fraction);
	chin = Point(shapes_face[0].part(8).x() / fraction, shapes_face[0].part(8).y() / fraction);
	top_nose = Point(shapes_face[0].part(27).x() / fraction, shapes_face[0].part(27).y() / fraction);
	symmetric_point = Point(chin.x * 2 - top_nose.x, chin.y * 2 - top_nose.y);
	upper_symmetric_point = Point(top_nose.x * 7 / 3 - chin.x * 4 / 3, top_nose.y * 7 / 3 - chin.y * 4 / 3);
}


//detect one side shoulder line
cv::vector<Point> MYcppGui::getFeatureFromUserInput(Mat shoulder_detection_image, Point head_shoulder, Point end_shoulder, int angle, int distance)
{
	LineIterator shoulder_sample(shoulder_detection_image, head_shoulder, end_shoulder, 8, false);
	//line(shoulder_detection_image, head_shoulder, end_shoulder, red, 3, 8, 0);

	cv::vector<Point> point_line;

	//Take points on shoulder_sample follow "distance" and build LineIterator from these point to symmetric_point
	for (int j = 0; j < shoulder_sample.count; j += distance) {

		int value_in_edge_map = 0;

		//LineIterator inside out. To get the inside point of user input.
		LineIterator it(shoulder_detection_image, symmetric_point, Point(shoulder_sample.pos()), 8, false);


		//Get all intersections of LineIterator and Canny line;
		for (int i = 0; i < it.count; i++, ++it)
		{
			Point current_point = Point(it.pos().x, it.pos().y);
			if (IsMatchToUserInput(current_point))
			{
				point_line.push_back(current_point); //point of it or point userInput should be add?
				circle(shoulder_detection_image, Point(it.pos().x, it.pos().y), 7, green, -1, 8);
				/*Mat feature = userInputFrame(Rect(current_point.x - 10, current_point.y - 10, 20, 20)).clone();
				cv::namedWindow("Shouldersample", CV_WINDOW_NORMAL);
				cv::resizeWindow("Shouldersample", 200, 200);
				cv::imshow("Shouldersample", feature);

				Mat detected_edges_feature;
				CannyProcessing(feature, detected_edges_feature);
				cv::namedWindow("CannyShouldersample", CV_WINDOW_NORMAL);
				cv::resizeWindow("CannyShouldersample", 200, 200);
				cv::imshow("CannyShouldersample", detected_edges_feature);*/

				waitKey(30);
				//featureCollection.push_back(feature);
				break;
			}
		}

		for (int i = 0; i < distance; i++)
		{
			shoulder_sample++;
		}
	}
	
	//cout << point_line << endl;
	//cout << point_line.size() << endl;
	return point_line;
}

//Show Sample Shoulder line
void MYcppGui::ShowSampleShoulder() {
	cv::namedWindow("Shouldersample", CV_WINDOW_NORMAL);
	cv::resizeWindow("Shouldersample", 200, 200);
	cv::imshow("Shouldersample", featureCollection[1]);
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

//void MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, Point head_shoulder, Point end_shoulder, int angle, int distance){
//
//	LineIterator upper_shoulder_sample(shoulder_detection_image, head_shoulder, end_shoulder, 8, false);
//	line(shoulder_detection_image, head_shoulder, end_shoulder, red, 3, 8, 0);
//
//	int length = 500;
//	double radian = angle * CV_PI / 180.0;
//
//	Point head_bottom_shoulder = chin;
//	Point end_bottom_shoulder = Point(head_bottom_shoulder.x + length*cos(radian), head_bottom_shoulder.y - length*sin(radian));
//	line(shoulder_detection_image, head_bottom_shoulder, end_bottom_shoulder, red, 3, 8, 0);
//
//	cv::vector<cv::vector<Point>> point_collection;
//	cv::vector<cv::vector<Point>> possible_lines;
//
//	//Take points on shoulder_sample follow "distance" and build LineIterator from these point to symmetric_point
//	for (int j = 0; j < upper_shoulder_sample.count; j += distance) {
//
//		int value_in_edge_map = 0;
//		int x_ = upper_shoulder_sample.pos().x;
//		int y_ = upper_shoulder_sample.pos().y;
//
//		//Find a start point for out checking line -> faster, save memory
//		if (upper_shoulder_sample.pos().x < 0)
//		{
//			x_ = 0;
//			y_ = FindY_LineEquationThroughTwoPoint(x_, upper_shoulder_sample.pos(), symmetric_point);
//		}
//
//		//new way
//		Point2f intersection_point;
//		intersection(head_bottom_shoulder, end_bottom_shoulder, upper_shoulder_sample.pos(), symmetric_point, intersection_point);
//
//		//Use the new start point 		//Go inside out
//		LineIterator it(shoulder_detection_image, intersection_point, Point(x_, y_), 8, false);
//		//line(shoulder_detection_image, intersection_point, Point(x_, y_), red, 3, 8, 0);
//
//		cv::vector<Point> point_line;
//
//		//Get all intersections of LineIterator and Canny line;
//		for (int i = 0; i < it.count; i++, ++it)
//		{
//			value_in_edge_map = detected_edges.at<uchar>(it.pos().y, it.pos().x);	// y first, x later
//			Point current_point = Point(it.pos().x, it.pos().y);
//			if (value_in_edge_map == 255)
//			{
//				if (point_line.empty() || ((abs(current_point.x - point_line.back().x) > 15) && (abs(current_point.y - point_line.back().y) > 15)))
//				{
//					circle(shoulder_detection_image, Point(it.pos().x, it.pos().y), 7, red, -1, 8);
//					point_line.push_back(Point(it.pos().x, it.pos().y));
//				}
//
//			}
//		}
//		point_collection.push_back(point_line);
//		for (int i = 0; i < distance; i++)
//		{
//			upper_shoulder_sample++;
//		}
//	}
//
//	//take potential point for shoulder line by checking angle of these line
//	for (int a = 0; a < point_collection.size() - 1; a++) {
//		for (int b1 = 0; b1 < point_collection[a].size(); b1++)
//		{
//			for (int b2 = 0; b2 < point_collection[a + 1].size(); b2++)
//			{
//				//Check difference of angle
//				cv::vector<Point> point_line;
//				if (abs(Angle(point_collection[a][b1], point_collection[a + 1][b2]) - angle) <= 20)
//				{
//					line(shoulder_detection_image, point_collection[a][b1], point_collection[a + 1][b2], red, 3, 8, 0);
//					point_line.push_back(point_collection[a][b1]);
//					point_line.push_back(point_collection[a + 1][b2]);
//				}
//			}
//		}
//	}
//
//
//	for (int i = 0; i < point_collection.size(); i++) {
//		for (int j = 0; j < point_collection[i].size(); j++) {
//			cv::vector<Point> path = findPath(j, i, point_collection, angle);
//
//			if (possible_lines.empty() || path.size() > possible_lines.back().size())
//			{
//				possible_lines.push_back(path);
//				//cout << "A new max line" << endl;
//			}
//			//cout << endl;
//		}
//	}
//
//	cv::vector<Point> shoulder_line = possible_lines.back();
//
//	for (int i = 0; i < shoulder_line.size() - 1; i++)
//	{
//		line(shoulder_detection_image, shoulder_line[i], shoulder_line[i + 1], blue, 3, 8, 0);
//	}
//
//}

//new fuction
void MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, bool leftHandSide, int angle, Scalar color, bool checkColor)
{
	//cv::namedWindow("Source", CV_WINDOW_NORMAL);
	//cv::resizeWindow("Source", 530, 700);

	double radian = angle * CV_PI / 180.0;
	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x); //used to be *2
	double length = abs(range_of_shoulder_sample / cos(radian));
	
	//bottom shoulder sample line
	Point head_bottom_shoulder = chin;
	Point end_bottom_shoulder = Point(head_bottom_shoulder.x + length*cos(radian), head_bottom_shoulder.y - length*sin(radian));
	Point end_second_bottom_shoulder = Point(end_bottom_shoulder.x, symmetric_point.y);
	line(shoulder_detection_image, head_bottom_shoulder, end_bottom_shoulder, red, 3, 8, 0);
	line(shoulder_detection_image, end_bottom_shoulder, end_second_bottom_shoulder, red, 3, 8, 0);

	//upper shoulder sample line
	Point head_upper_shoulder;
	if (leftHandSide)
	{
		head_upper_shoulder = Point(left_cheek.x - distance_from_face_to_shouldersample, left_cheek.y);
	}
	else 
	{
		head_upper_shoulder = Point(right_cheek.x + distance_from_face_to_shouldersample, right_cheek.y);
	}
	
	Point end_upper_shoulder = Point(head_upper_shoulder.x + length*2.5*cos(radian), head_upper_shoulder.y - length*2.5*sin(radian));
	line(shoulder_detection_image, head_upper_shoulder, end_upper_shoulder, red, 3, 8, 0);
	

	cv::vector<cv::vector<Point>> point_collection;
	
	//intersection_point_01 to mark where the second bottom shoulder line start
	Point2f intersection_point_01;
	intersection(head_upper_shoulder, end_upper_shoulder, symmetric_point, end_bottom_shoulder, intersection_point_01);

	//Take points on shoulder_sample follow "checking_block" and build LineIterator from these point to symmetric_point
	for (int j = 0; abs(checking_block*j*cos(radian)) < range_of_shoulder_sample*2.5; j++) {
		Point current_point = Point(head_upper_shoulder.x + checking_block*j*cos(radian), head_upper_shoulder.y - checking_block*j*sin(radian));
		int value_in_edge_map = 0;
		Point2f intersection_point;

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

		//Use the new start point 		//Go inside out
		LineIterator it(shoulder_detection_image, intersection_point, current_point, 8, false);
		
		//line(shoulder_detection_image, intersection_point, current_point, red, 3, 8, 0);
		//cv::imshow("Source", shoulder_detection_image);
		//waitKey(300);

		cv::vector<Point> point_line;

		//Get all intersections of LineIterator and Canny line;
		for (int i = 0; i < it.count; i+=2, ++it, ++it)
		{
			value_in_edge_map = detected_edges.at<uchar>(it.pos().y, it.pos().x);	// y first, x later
			Point current_point = Point(it.pos().x, it.pos().y);
			
			//problem only on train_10
			if (it.pos().y + 25 >= shoulder_detection_image.size().height)
				break;
			Vec3b color = shoulder_detection_image.at<Vec3b>(Point(it.pos().x, it.pos().y + 25));

			bool is_match_color = IsMatchToColorCollectionInput(color);
			if (!checkColor)
				is_match_color = true;

			if (value_in_edge_map == 255 && is_match_color)
			{
				if (point_line.empty() || EuclideanDistance(current_point, point_line.back()) >= 15)	//10 work really well - 15 works well too
				{
					circle(shoulder_detection_image, Point(it.pos().x, it.pos().y), 5, red, -1, 8);
					point_line.push_back(Point(it.pos().x, it.pos().y));
				}
			}
		}
		point_collection.push_back(point_line);
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
	cv::vector<cv::vector<Point>> possible_lines;
	cv::vector<cv::vector<Point>> possible_lines_for_arm;


	for (int i = 0; i < point_collection.size(); i++) {
		for (int j = 0; j < point_collection[i].size(); j++) {
			cv::vector<Point> path = findPath(j, i, point_collection, angle);
			cv::vector<Point> path_for_arm = findPath(j, i, point_collection, angle_for_arm);

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
	cv::vector<Point> shoulder_line_for_arm_longest = possible_lines_for_arm.back();

	for (int i = 0; i < shoulder_line_for_arm_longest.size() - 1; i++)
	{
		line(shoulder_detection_image, shoulder_line_for_arm_longest[i], shoulder_line_for_arm_longest[i + 1], black, 5, 8, 0);
	}


	cv::vector<Point> shoulder_line = possible_lines.back();

	for (int i = 0; i < shoulder_line.size() - 1; i++)
	{
		line(shoulder_detection_image, shoulder_line[i], shoulder_line[i + 1], color, 5, 8, 0);
	}


	//new way which is to detect the position of arm line
	//list use pushback, so the last one is [0]
	int index_one_third = shoulder_line.size() * 1 / 3;
	int index_half = shoulder_line.size()/ 2;
	cv::vector<Point> shoulder_line_for_arm_test;

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
			line(shoulder_detection_image, shoulder_line_for_arm_test[i], shoulder_line_for_arm_test[i + 1], color, 5, 8, 0);
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

void MYcppGui::ImageProcessing_WithUserInput(Mat &frame, bool isTesting) {
	
	
	Mat src, face_detection_frame; // change name soon
	src = frame.clone(); // Use for Canny
	face_detection_frame = frame.clone(); // Use for shoulder detection

	if (userInputFrame.empty())
	{
		userInputFrame = frame.clone();
		//-------------------------collect sample color of shouder--------------------
		collectColorShoulder();
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

	//----------------------------- Deal with userInput - start --------------------
	
	//for (int i = 0; i < userInput.size(); i++)
	//{
	//	circle(face_detection_frame, userInput[i], 7, blue, -1, 8);
	//}
	//cv::imshow("Source", face_detection_frame);

	//----------------------------- Detect face - start ---------------------
	double fraction = 1;
	std::vector<dlib::full_object_detection> shapes_face;

	
	//shapes_face = face_detection_dlib_image(face_detection_frame);
	shapes_face = face_detection_update(face_detection_frame);
	
	

	//No face is detected
	if (shapes_face.size() == 0)
	{
		return;
	}

	for (int i = 0; i < 6; i++)
	{
		line(face_detection_frame, Point(shapes_face[0].part(i).x() / fraction, shapes_face[0].part(i).y() / fraction), 
			Point(shapes_face[0].part(i + 1).x() / fraction, shapes_face[0].part(i + 1).y() / fraction), green, 5, 8, 0);
	}

	for (int i = 11; i < 16; i++)
	{
		line(face_detection_frame, Point(shapes_face[0].part(i).x() / fraction, shapes_face[0].part(i).y() / fraction),
			Point(shapes_face[0].part(i + 1).x() / fraction, shapes_face[0].part(i + 1).y() / fraction), green, 5, 8, 0);
	}

	detectNecessaryPointsOfFace(shapes_face);

	circle(face_detection_frame, left_cheek, 5, green, -1, 8);
	circle(face_detection_frame, right_cheek, 5, green, -1, 8);
	circle(face_detection_frame, top_nose, 5, green, -1, 8);
	circle(face_detection_frame, chin, 5, green, -1, 8);
	circle(face_detection_frame, symmetric_point, 5, green, -1, 8);
	circle(face_detection_frame, upper_symmetric_point, 5, green, -1, 8);

	double left_eye_width = shapes_face[0].part(39).x() - shapes_face[0].part(36).x();

	if (distance_from_face_to_shouldersample == 0 || abs(left_eye_width * 3 / 4 - distance_from_face_to_shouldersample) >= 50)
	{
		distance_from_face_to_shouldersample = left_eye_width * 3 / 4;
	}

	if (checking_block == 0 || abs(left_eye_width * 1 / 3 - checking_block) >= 50)
	{
		checking_block = left_eye_width * 1.3 / 2;
	}

	
	//-----------------------------	Preprocess a part of image to speed up ---------------------------
	clock_t tmp01 = clock();

	double range_of_shoulder_sample = (right_cheek.x - left_cheek.x);

	Point pA = Point(max(left_cheek.x - int(range_of_shoulder_sample*2.5), 0), min(left_cheek.y, right_cheek.y));
	Point pB = Point(min(right_cheek.x + int(range_of_shoulder_sample*2.5), frame.cols), pA.y);
	Point pC = Point(pB.x, min(symmetric_point.y, frame.rows));
	Point pD = Point(pA.x, pC.y);

	Mat sub_frame = src(cv::Rect(pA.x, pA.y, pB.x - pA.x, pD.y - pA.y));	
	Mat CannyWithoutBlurAndMorphology = Preprocessing(sub_frame);
	
	//Add preprocessed part to a frame that is in the same size with the old one
	Mat BiggerCannyWithoutBlurAndMorphology(frame.rows, frame.cols, CV_8UC1, Scalar(0));
	CannyWithoutBlurAndMorphology.copyTo(BiggerCannyWithoutBlurAndMorphology(cv::Rect(pA.x, pA.y, CannyWithoutBlurAndMorphology.cols, CannyWithoutBlurAndMorphology.rows)));

	clock_t tmp02 = clock();
	std::cout << " Time for Preprocess: " << float(tmp02 - tmp01) / CLOCKS_PER_SEC << endl;
	

	//-----------------------------shoulders---------------------------
	double angle_left = -150;
	int angle_right = -30;
	Mat face_detection_frame_Blur_Check = face_detection_frame.clone();
	Mat face_detection_frame_Blur_NoCheck = face_detection_frame.clone();

	detectShoulderLine(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, true, angle_left, green, true);
	detectShoulderLine(face_detection_frame, BiggerCannyWithoutBlurAndMorphology, false, angle_right, green, true);

	//-----------------------------testing shoulder---------------------------
	if (isTesting) {
		detectShoulderLine(face_detection_frame_Blur_NoCheck, detected_edges, true, angle_left, blue, false);
		detectShoulderLine(face_detection_frame_Blur_Check, detected_edges, true, angle_left, blue, true);

		detectShoulderLine(face_detection_frame_Blur_NoCheck, detected_edges, false, angle_right, blue, false);
		detectShoulderLine(face_detection_frame_Blur_Check, detected_edges, false, angle_right, blue, true);

		cv::namedWindow("Blur_Check", CV_WINDOW_NORMAL);
		cv::resizeWindow("Blur_Check", 530, 700);
		cv::namedWindow("Blur_NoCheck", CV_WINDOW_NORMAL);
		cv::resizeWindow("Blur_NoCheck", 530, 700);

		cv::imshow("Blur_Check", face_detection_frame_Blur_Check);
		cv::imshow("Blur_NoCheck", face_detection_frame_Blur_NoCheck);

		cv::namedWindow("Source_NoBlur_Check", CV_WINDOW_NORMAL);
		cv::resizeWindow("Source_NoBlur_Check", 530, 700);
		cv::imshow("Source_NoBlur_Check", face_detection_frame);
		cv::imshow("Canny Only", BiggerCannyWithoutBlurAndMorphology);
	}	
	
	//return face_detection_frame
	frame = Mat(face_detection_frame);

	std::cout << " Time for Postprocess: " << float(clock() - tmp02) / CLOCKS_PER_SEC << endl;
}

void MYcppGui::collectColorShoulder()
{
	for (int i = 0; i < userInput.size(); i+=5)
	{
		int x = userInput[i].x;
		int y = userInput[i].y + 25;
		int color_value = userInputFrame.at<uchar>(Point(x, y));
		Vec3b color = userInputFrame.at<Vec3b>(Point(x, y));
		
		colorValueCollection.push_back(color_value);
		
		if (i == 0) 
		{
			colorCollection.push_back(color);
			continue;
		}

		double minColourDistance = ColourDistance(color, colorCollection[0]);
		cout << minColourDistance << endl;

		for (int j = 1; j < colorCollection.size(); j++)
		{
			double colourDistance = ColourDistance(color, colorCollection[j]);
			cout << ColourDistance(color, colorCollection[j]) << endl;
			if (colourDistance < minColourDistance)
				minColourDistance = colourDistance;
		}

		circle(userInputFrame, Point(x, y), 5, green, -1, 8);

		if (minColourDistance > 30)
		{
			colorCollection.push_back(color);
			circle(userInputFrame, Point(x, y), 5, blue, -1, 8);
		}

		cout << "-----------------" << endl;
	
		
	}
}

bool MYcppGui::IsMatchToColorCollectionInput(Vec3b color)
{
	for (int i = 0; i < colorCollection.size(); ++i)
	{
		if (ColourDistance(color, colorCollection[i]) <= 30){
			return true;
		}
	}
	return false;
}
double EuclideanDistance(Point p1, Point p2) {
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)); 
}

double ColourDistance(Vec3b e1, Vec3b e2)
{
	//0 - blue 
	//1 - green
	//2 - red
	long rmean = ((long)e1[2] + (long)e2[2]) / 2;
	long r = (long)e1[2] - (long)e2[2];
	long g = (long)e1[1] - (long)e2[1];
    long b = (long)e1[0] - (long)e2[0];
    return sqrt((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8));
}

double Angle(Point start, Point end)
{
	const double Rad2Deg = 180.0 / CV_PI;
	const double Deg2Rad = CV_PI / 180.0;
	return atan2(start.y - end.y, end.x - start.x) * Rad2Deg;
}

cv::vector<Point> MYcppGui::findPath(int index, int index_line, cv::vector<cv::vector<Point>> point_collection, double angle) {
	//cout << index << " " << index_line << " " << point_collection.size() << endl;
	if (index_line >= point_collection.size() - 1) {
		cv::vector<Point> result;
		result.push_back(point_collection[index_line][index]);
		return result;
	}

	cv::vector<Point> new_point_line;
	cv::vector<Point> tmp_new_point_line;

	for (int i = 0; i < point_collection[index_line + 1].size(); i++)
	{
		//check angle
		if (abs(Angle(point_collection[index_line][index], point_collection[index_line + 1][i]) - angle) <= 25) {
			tmp_new_point_line = findPath(i, index_line + 1, point_collection, angle);
		}
		tmp_new_point_line.push_back(point_collection[index_line][index]);

		if (new_point_line.empty() || tmp_new_point_line.size() > new_point_line.size())
		{
			new_point_line = tmp_new_point_line;
		}
		tmp_new_point_line.clear();
	}
	return new_point_line;
}

//check if the point is belong to userInput.
bool MYcppGui::IsMatchToUserInput(Point point)
{
	for (int i = 0; i < userInput.size(); i++)
	{
		if (abs(point.x - userInput[i].x) < 3 && abs(point.y - userInput[i].y) < 3)
		{
			return true;
		}
	}

	return false;
}

float FindY_LineEquationThroughTwoPoint(float x_, Point p1, Point p2)
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
