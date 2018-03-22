#pragma once
#include "cppOpencv.h"
#include <sstream>
#include <string>


int main() {
	MYcppGui *myGui = new MYcppGui();
	vector<vector<Point2f>> userInput;
	Mat frame;
	Mat src;
	string pathUserInput = "D:\\605\\Source code\\UserInput\\input_";
	string pathData = "D:\\605\\Source code\\dataset\\train_";
	
	int n;
	bool image_version = true;
	bool all = false;
	bool saveOnly = false;

	cout << "| 0: Image | 1: Video | 2: All Images " << endl 
		<< "| 3: Some Images | 4: Read image from file |" << endl 
		<< "| 5: All videos | 6: Some videos |:";
	cin >> n;
	if (n == 0)
		image_version = true;
	if (n == 1)
		image_version = false;
	if (n == 2 || n == 3 || n == 5 || n == 6)
		all = true;
	if (n == 5 || n == 6){
		image_version = false;
		saveOnly = true;
	}

	if (n == 4) {
		string file_path = "D:\\605\\Source code\\dataset\\file_";
		int fileNo;
		//cout << "File name: ";
		//cin >> fileNo;
		fileNo = 1;
		string name = to_string(fileNo);
		for (int i = name.size(); i < 2; i++) {
			name = "0" + name;
		}

		file_path = file_path + name + ".txt";

		std::ifstream infile(file_path);
		std::string line;
		if (infile.good()){
			while (std::getline(infile, line))
			{
				std::istringstream iss(line);
				int a;
				if (!(iss >> a )) { break; } 

				string n = to_string(a);
				for (int i = n.size(); i < 3; i++) {
					n = "0" + n;
				}

				string path = pathData + n + ".jpg";
				frame = cv::imread(path, CV_LOAD_IMAGE_COLOR);
				if (!frame.data) {
					continue;
				}

				string tmpPath;
				tmpPath = pathUserInput + n + ".txt";

				MYcppGui *myGui = new MYcppGui();
				myGui->AddUserInput(tmpPath);

				
				myGui->FILE_NAME = n;

				cout << n << endl;
				vector<Mat> resultMats = myGui->ImageProcessing_Final(frame, false, true, true);
				imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\xx\\result_" + n + ".jpg", frame);
				for (int j = 0; j < resultMats.size(); j++) {
					imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\xx\\result_" + n + "(1).jpg", resultMats[j]);
				}
			}
		}
		return 0;
	}

	if (all) {
		int from = 2, to = 400;

		if (n == 3 || n == 6) {
			cout << "From: ";
			cin >> from;
			cout << "To: ";
			cin >> to;
		}
		for (int i = from; i <= to; i++) {
			string fileName = to_string(i);
			cout << " --- " << fileName << " --- " << endl;
			try {
				vector<Mat> resultMats = myGui->TotalProcess(fileName, image_version, saveOnly);
				//if (!frame.data) {
				//	continue;
				//}
				//imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\1\\result_" + n + ".jpg", frame);

				for (int j = 0; j < resultMats.size() && image_version; j++) {
					imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\1\\result_" + fileName + "(" + to_string(j) + ").jpg", resultMats[j]);
				}
			}
			catch (std::out_of_range& exc) {
				std::cerr << exc.what();
				cout << "Error!" << endl;
				continue;
			}
			catch (...) {
				cout << "Error!" << endl;
				continue;
			}
		}

		cout << "DONE ALL!" << endl;
	}
	else {
		string fileName;
		cout << "File name: ";
		cin.ignore();
		getline(cin, fileName);

		if (fileName == "")
			fileName = "002";

		myGui->TotalProcess(fileName, image_version, saveOnly);
		waitKey(0);
	}
	return 0;
}

