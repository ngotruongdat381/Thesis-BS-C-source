#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace std;
using namespace cv;
//using namespace dlib;

const clock_t begin_time = clock();
static clock_t current_time = clock();

enum Status { Walking, Appearing, Disappearing };


const Scalar blue = Scalar(255, 0, 0);
const Scalar green = Scalar(0, 255, 0);
const Scalar red = Scalar(0, 0, 255);
const Scalar black = Scalar(0, 0, 0);
const Scalar yellow = Scalar(0, 255, 255);
const Scalar organe = Scalar(243, 97, 53);
const Scalar white = Scalar(255, 225, 255);

const int RIGHT_LINE = 1;
const int LEFT_LINE = 0;

const int RIGHT = 1;
const int LEFT = -1;

static int FACE_DOWNSAMPLE_RATIO = 3;
static int SKIP_FRAMES = 3;

double EuclideanDistance(Point2f p1, Point2f p2);
double ColourDistance(Vec3b e1, Vec3b e2);
double ColourDistance_LAB(Vec3f e1, Vec3f e2);

string GetTime();
double Angle(Point2f start, Point2f end);
float FindY_LineEquationThroughTwoPoint(float x_, Point2f p1, Point2f p2);
bool isSegmentsIntersecting(Point2f& p1, Point2f& p2, Point2f& q1, Point2f& q2);
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);
Point2f mirror(Point2f p, Point2f point0, Point2f point1);

class MYcppGui {
public:
	MYcppGui();
	~MYcppGui();

	//Not good
	//void collectColorShoulder_LAB();
	//bool IsMatchToColorCollectionInput_LAB(Vec3f color_LAB);

	Mat GetThumnail(string fileName);
	int myCppLoadAndShowRGB(string fileName);
	void MYcppGui::VideoProcessing(string fileName);
	//void MYcppGui::ImageProcessing(Mat &frame);
	vector<Mat> MYcppGui::ImageProcessing_Final(Mat &frame, bool withUserInput, bool isTesting, bool DebugLine);
	Mat MYcppGui::ImageProcessing(string fileName, vector<cv::Point2f> userInput);

	void MYcppGui::Morphology_Operations(Mat &src);
	void MYcppGui::CannyProcessing(Mat image, OutputArray edges);
	std::vector<dlib::full_object_detection> MYcppGui::face_detection_dlib_image(Mat frame);
	std::vector<dlib::full_object_detection> MYcppGui::face_detection_update(Mat frame);
	void MYcppGui::detectNecessaryPointsOfFace(std::vector<dlib::full_object_detection> shapes_face);
	void MYcppGui::CorrectFaceDetection(std::vector<dlib::full_object_detection>& shapes_face, Mat &mask_skin);

	//void MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, Point head_shoulder, Point end_shoulder, int angle, int distance);
	cv::vector<Point2f> MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, bool leftHandSide, int angle, Scalar color
										, bool checkColor, bool checkPreviousResult);
	void AddSticker(Mat &frame);

	cv::vector<Point2f> DetectNeckLines(Mat shoulder_detection_image, Mat detected_edges, Mat mask_skin, std::vector<dlib::full_object_detection> shapes_face,
		bool leftHandSide, int angle_neck);

	cv::vector<Point2f> findPath(int index, int index_line, cv::vector<cv::vector<Point2f>> point_collection, double angle);
	void AddUserInput(string path);
	vector<vector<Point2f>> readUserInput(string path);

	bool MYcppGui::IsMatchToUserInput(Point2f point);
	bool IsMatchToColorCollectionInput(Vec3b color);
	bool MYcppGui::IsMatchColor(Vec3b color, Vector<Vec3b> Collection, int epsilon);

	void collectColorShoulderFromInput();
	void MYcppGui::collectColor(Mat&frame, vector<Vec3b> &colorCollection, Point2f head_point, Point2f end_point, double epsilon);
	void MYcppGui::collectColorShoulder(Mat& frame);
	Vector<Vec3b> MYcppGui::collectColorNeck(Mat&frame, Point2f head_neck, Point2f end_neck);
	Mat Preprocessing(Mat frame);
	void GetSticker(string name, bool changeDirection);
	Mat MYcppGui::RemoveUnneccessaryImage(Mat& frame);
	void RefinePoint_collection(Mat& frame, cv::vector<cv::vector<Point2f>> &point_collection);

	vector<double> CompareToGroundTruth(vector<vector<Point2f>> line);
	double MYcppGui::OverlapPercentage(vector<Point2f> groundTruth, vector<Point2f> line);
private:
	dlib::shape_predictor shape_predictor;
	Mat userInputFrame;
	vector<vector<Point2f>> userInput;
	vector<vector<Point2f>> simplifizedUserInput;
	vector<vector<Point2f>> current_shoulderLine;
	vector<vector<Point2f>> simplifized_current_shoulderLine;

	vector<Vec3b> colorCollection_Shoulder;
	//vector<Vec3f> colorCollection_LAB;

	double checking_block;
	int nth = 1;	//nth frame
	std::vector<dlib::rectangle> cur_dets;
	
	//stickers variable
	string stickerName = "pokemon";
	Vector<Mat> stickerFrames;
	int index_stickerFrames = 0;
	double relativePostion_sticker;
	double x_ROI_sticker_begin = 0;
	bool in_cropping_process = false;
	Status stickerStatus = Walking;
	bool Disappeared = false;
	int stickerDirection = LEFT;

	Point2f left_cheek = NULL;
	Point2f right_cheek = NULL;
	Point2f chin = NULL;
	Point2f top_nose = NULL;
	Point2f nose = NULL;
	Point2f symmetric_point = NULL;
	Point2f upper_symmetric_point = NULL;

	bool TEST_MODE = true;
	bool STICKER_MODE = false;
	bool TRACKING_MODE = false;
};