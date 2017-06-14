#include "cppOpencv.h"
#include <sstream>
#include <string>

vector<vector<Point>> readUserInput(string path)
{
	vector<cv::Point> leftShoulderInput;
	vector<cv::Point> rightShoulderInput;
	vector<vector<Point> > inputs;

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
			leftShoulderInput.push_back(Point(a, b));
		}
		else {
			rightShoulderInput.push_back(Point(a, b));
		}
		previous_a = a;
	}
	inputs.push_back(leftShoulderInput);
	inputs.push_back(rightShoulderInput);

	return inputs;
}

int main() {
	vector<vector<Point>> userInput;
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
		myGui->ImageProcessing_WithUserInput(frame, true);
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