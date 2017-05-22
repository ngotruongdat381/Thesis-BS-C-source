#include "cppOpencv.h"
#include <sstream>
#include <string>

vector<cv::Point> readUserInput(string path)
{
	vector<cv::Point> userInput;
	std::ifstream infile(path);
	std::string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		double a, b;
		if (!(iss >> a >> b)) { break; } // error
		userInput.push_back(Point(a, b));
	}
	return userInput;
}

int main() {
	vector<cv::Point> userInput;
	Mat frame;
	Mat src;
	string pathUserInput = "D:\\605\\Source code\\UserInput\\input_";
	string pathData = "D:\\605\\Source code\\dataset\\train_";
	
	int n;
	bool image_version;

	cout << "0: Image | 1: Video : ";
	cin >> n;
	if (n = 0)
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
	if (image_version)
	{
		pathData = pathData + fileName + ".jpg";
		frame = cv::imread(pathData, CV_LOAD_IMAGE_COLOR);
		myGui->ImageProcessing_WithUserInput(frame, true);
	}
	//for video version
	else	
	{
		pathData = pathData + "video_" + fileName + ".avi";
		myGui->VideoProcessing(pathData);
	}

	//cv::namedWindow("Sourcez", CV_WINDOW_NORMAL);
	//cv::resizeWindow("Sourcez", 282, 502);
	//cv::imshow("Sourcez", frame);

	//myGui->ShowSampleShoulder();


	waitKey(0);
	return 0;
}