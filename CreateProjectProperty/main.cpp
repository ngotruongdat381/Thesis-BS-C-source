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
	
	string fileName;
	cout << "File name: ";
	getline(cin, fileName);

	if (fileName == "")
		fileName = "02";

	pathUserInput = pathUserInput + fileName + ".txt";
	pathData = pathData + fileName + ".jpg";


	userInput = readUserInput(pathUserInput);
	MYcppGui *myGui = new MYcppGui();
	myGui->AddUserInput(userInput);

	//Load an image
	frame = cv::imread(pathData, CV_LOAD_IMAGE_COLOR);
	myGui->ImageProcessing_WithUserInput(frame);
	
	//myGui->VideoProcessing(".\\dataset\\test02.avi");

	//frame = myGui->ImageProcessing(".\\dataset\\train_02.jpg");
	
	//cv::namedWindow("Sourcez", CV_WINDOW_NORMAL);
	//cv::resizeWindow("Sourcez", 282, 502);
	//cv::imshow("Sourcez", frame);

	//myGui->ShowSampleShoulder();


	waitKey(0);
	return 0;
}