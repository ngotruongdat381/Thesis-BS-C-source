#include "cppOpencv.h"
#include <sstream>
#include <string>

vector<vector<Point2f>> readUserInput(string path)
{
	vector<cv::Point2f> leftShoulderInput;
	vector<cv::Point2f> rightShoulderInput;
	vector<vector<Point2f> > inputs;

	std::ifstream infile(path);
	std::string line;
	double previous_a = 0;
	bool left = true;

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

	return inputs;
}

int main() {
	vector<vector<Point2f>> userInput;
	Mat frame;
	Mat src;
	string pathUserInput = "D:\\605\\Source code\\UserInput\\input_";
	string pathData = "D:\\605\\Source code\\dataset\\train_";
	
	int n;
	bool image_version;

	cout << "0: Image | 1: Video : ";
	cin >> n;
	if (n == 0)
		image_version = true;
	else
		image_version = false;

	string fileName;
	cout << "File name: ";
	cin.ignore();
	getline(cin, fileName);

	if (fileName == "")
		fileName = "02";

	pathUserInput = pathUserInput + fileName + ".txt";
	userInput = readUserInput(pathUserInput);
	MYcppGui *myGui = new MYcppGui();
	myGui->AddUserInput(userInput);

	//for image
	if (image_version) {
		pathData = pathData + fileName + ".jpg";
		frame = cv::imread(pathData, CV_LOAD_IMAGE_COLOR);
		myGui->ImageProcessing_WithUserInput(frame, true, true);
	}
	//for video version
	else {
		pathData = pathData + "video_" + fileName + ".avi";
		myGui->VideoProcessing(pathData);
	}

	//myGui->ShowSampleShoulder();

	waitKey(0);
	return 0;
}