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
const Scalar yellow = Scalar(0, 255, 255);
const Scalar purple = Scalar(255, 0, 255);
const Scalar sky = Scalar(255, 255, 0);
const Scalar black = Scalar(0, 0, 0);
const Scalar white = Scalar(255, 225, 255);

const vector<Scalar> colors{ blue, green, red, yellow, purple, sky, black, white };
const int RIGHT_LINE = 1;
const int LEFT_LINE = 0;

const int RIGHT = 1;
const int LEFT = -1;

const double LEFT_SHOULDER_ANGLE = -150;
const double LEFT_ARM_ANGLE = -110;
const double RIGHT_SHOULDER_ANGLE = -30;
const double RIGHT_ARM_ANGLE = -70;

static int FACE_DOWNSAMPLE_RATIO = 3;
static int SKIP_FRAMES = 3;

static bool TEST_MODE = true;
static bool STICKER_MODE = true;
static bool TRACKING_MODE = false;
static bool VIDEO_MODE = false;
static bool EXPERIMENT_MODE = false;


static double num_of_processed_image;
static Vector<double> percentageOverlapDatas;

struct Experiment {
	double number = 0;
	vector<double> face_detection_cost;
	vector<double> preprocess_cost;
	vector<double> color_collection_cost;
	vector<double> shoulder_detection_cost;
	vector<double> total_cost;
};

extern vector<Experiment> Resolutions;
extern int current_px;



struct PointPostion {
	int index_line;
	int index;
	PointPostion();
	PointPostion(int index_line_, int index_) {
		index_line = index_line_;
		index = index_;
	}
	Point2f getFrom(vector<vector<Point2f>> point_collection) {
		if (point_collection.size() > index_line && point_collection[index_line].size() > index) {
			return point_collection[index_line][index];
		}
		return Point();
	}
};

struct ShoulderModel{
	int angle;
	double radian;
	double range_of_shoulder_sample;
	double length;
	Point2f head_bottom_shoulder;
	Point2f end_bottom_shoulder;
	Point2f end_second_bottom_shoulder;
	Point2f head_upper_shoulder;
	Point2f end_upper_shoulder;
	bool leftHandSide;

	ShoulderModel();
	ShoulderModel(int angle_, double range_of_shoulder_sample_, Point2f head_bottom_shoulder_, Point2f end_second_bottom_shoulder_, Point2f head_upper_shoulder_){
		angle = angle_;
		radian = angle * CV_PI / 180.0;
		range_of_shoulder_sample = range_of_shoulder_sample_;
		length = abs(range_of_shoulder_sample / cos(radian));
		
		head_bottom_shoulder = head_bottom_shoulder_;
		end_bottom_shoulder = Point2f(head_bottom_shoulder.x + length*cos(radian), head_bottom_shoulder.y - length*sin(radian));
		end_second_bottom_shoulder = end_second_bottom_shoulder_;
		
		head_upper_shoulder = head_upper_shoulder_;
		end_upper_shoulder = Point2f(head_upper_shoulder.x + length*2.5*cos(radian), head_upper_shoulder.y - length*2.5*sin(radian));
	}
	
	ShoulderModel(int angle_, double radian_, double range_of_shoulder_sample_, double length_, Point2f head_bottom_shoulder_, Point2f end_bottom_shoulder_, Point2f end_second_bottom_shoulder_, Point2f head_upper_shoulder_, Point2f end_upper_shoulder_, bool leftHandSide_){
		angle = angle_;
		radian = radian_;
		range_of_shoulder_sample = range_of_shoulder_sample_;
		length = length_;
		head_bottom_shoulder = head_bottom_shoulder_;
		end_bottom_shoulder = end_bottom_shoulder_;
		end_second_bottom_shoulder = end_second_bottom_shoulder_;
		head_upper_shoulder = head_upper_shoulder_;
		end_upper_shoulder = end_upper_shoulder_;
		leftHandSide = leftHandSide_;
	}
};

class MYcppGui {
public:
	string FILE_NAME;
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
	vector<Point2f> MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, bool leftHandSide, int angle, Scalar color
										, bool checkColor, bool checkPreviousResult);
	vector<vector<Point2f>> Collect_Potential_ShoulderPoint(Mat shoulder_detection_image, Mat &detected_edges, ShoulderModel &shoulderModel, bool checkColor, bool checkPreviousResult, Scalar color);
	vector<Point2f> Finding_ShoulderLines_From_PointCollection(Mat shoulder_detection_image, vector<vector<Point2f>> point_collection, ShoulderModel &shoulderModel, Scalar color, double epsilon_angle_shoulder);
	vector<Point2f> Finding_ShoulderLines_From_PointCollection_v2(Mat shoulder_detection_image, vector<vector<Point2f>> point_collection, ShoulderModel &shoulderModel, Scalar color, double epsilon_angle_shoulder);
	vector<Point2f> Finding_ShoulderLines_From_PointCollection_v3(Mat shoulder_detection_image, vector<vector<Point2f>> point_collection, bool leftHandSide, int angle, Scalar color);
	
	void Improve_Fail_Detected_ShoulderLine(Mat &shoulder_detection_image, vector<Point2f> &shoulder_line, Point2f head_upper_shoulder, int angle);
	void MYcppGui::Refine_Overlap_ShoulderLine_And_ArmLine(Mat &shoulder_detection_image, vector<Point2f> &shoulder_line, vector<Point2f> &shoulder_line_for_arm_longest);

	void AddSticker(Mat &frame);

	vector<Point2f> DetectNeckLines(Mat shoulder_detection_image, Mat detected_edges, Mat mask_skin, std::vector<dlib::full_object_detection> shapes_face,
		bool leftHandSide, int angle_neck);

	vector<Point2f> findPath(int index_line, int index, vector<vector<Point2f>> &point_collection, double angle, double epsilon, int type);

	//type: 0: Shoulder - 1: Arm - 2: Neck 
	vector<PointPostion> findPath_v2(int index_line, int index, vector<vector<Point2f>> &point_collection, double angle, double epsilon, int type, ShoulderModel &shoulderModel);
	vector<vector<PointPostion>> MYcppGui::findPath_new(int index_line, int index, vector<vector<Point2f>> &point_collection, double angle);
	void AddUserInput(string path);
	vector<vector<Point2f>> readUserInput(string path);

	bool MYcppGui::IsMatchToUserInput(Point2f point);
	bool IsMatchToColorCollectionInput(Vec3b color);
	bool MYcppGui::IsMatchColor(Vec3b color, Vector<Vec3b> Collection, int epsilon);

	void collectColorShoulderFromInput(Mat &frame);
	void MYcppGui::collectColor(Mat&frame, Mat &mask_skin, vector<Vec3b> &colorCollection, Point2f head_point, Point2f end_point, double epsilon, bool splitSkinColor);
	void MYcppGui::collectColorShoulder(Mat& frame, Mat &mask_skin, bool splitSkinColor);
	Vector<Vec3b> MYcppGui::collectColorNeck(Mat&frame, Point2f head_neck, Point2f end_neck);
	Mat Preprocessing(Mat frame);
	Mat StrongPreprocessing(Mat frame);

	void GetSticker(string name, bool changeDirection);
	Mat MYcppGui::RemoveUnneccessaryImage(Mat& frame);
	void RefinePoint_collection(Mat& frame, vector<vector<Point2f>> &point_collection);
	int RefinePoint_collection_v2(Mat& shoulder_detection_image, Mat &detected_edges, vector<vector<Point2f>> &point_collection, ShoulderModel &shoulderModel);
	void MYcppGui::RefinePoint_collection_v21(Mat& frame, vector<vector<Point2f>> &point_collection);
	void MYcppGui::RefinePoint_collection_v22(Mat& frame, vector<vector<Point2f>> &point_collection);
	void MYcppGui::RefinePoint_collection_v23(Mat& frame, vector<vector<Point2f>> &point_collection, ShoulderModel &shoulderModel);
	double Check_Density(Mat& shoulder_detection_image, Mat &detected_edges, Point2f head_bottom_point, int angle, double height);

	vector<double> CompareToGroundTruth(vector<vector<Point2f>> line);
	double MYcppGui::OverlapPercentage(vector<Point2f> groundTruth, vector<Point2f> line);

private:
	dlib::shape_predictor shape_predictor;
	Mat userInputFrame;
	Mat originalFrame;
	vector<vector<Point2f>> userInput;
	vector<vector<Point2f>> simplifizedUserInput;
	vector<vector<Point2f>> current_shoulderLine;
	vector<vector<Point2f>> simplifized_current_shoulderLine;

	vector<Vec3b> colorCollection_Shoulder;
	vector<Vec3b> colorCollection_Skin;
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

	//Key point
	Point2f left_cheek = NULL;
	Point2f right_cheek = NULL;
	Point2f chin = NULL;
	Point2f top_nose = NULL;
	Point2f nose = NULL;
	Point2f symmetric_point = NULL;
	Point2f upper_symmetric_point = NULL;
};

//General function
double EuclideanDistance(Point2f p1, Point2f p2);
double ColourDistance(Vec3b e1, Vec3b e2);
double ColourDistance_LAB(Vec3f e1, Vec3f e2);

string GetTime();
double Angle(Point2f start, Point2f end);
double AngleDifference(double angleA, double angleB);
float FindY_LineEquationThroughTwoPoint(float x_, Point2f p1, Point2f p2);
bool isSegmentsIntersecting(Point2f& p1, Point2f& p2, Point2f& q1, Point2f& q2);
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);
Point2f mirror(Point2f p, Point2f point0, Point2f point1);
vector<Point2f> ConvertFromMap(vector<vector<Point2f>> point_collection, vector<PointPostion> map);
