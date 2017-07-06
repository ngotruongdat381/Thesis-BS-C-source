#include "cppOpencv.h"
#include <sstream>
#include <string>


int main() {
	vector<vector<Point2f>> userInput;
	Mat frame;
	Mat src;
	string pathUserInput = "D:\\605\\Source code\\UserInput\\input_";
	string pathData = "D:\\605\\Source code\\dataset\\train_";
	
	int n;
	bool image_version = true;
	bool all = false;

	cout << "| 0: Image | 1: Video | 2: All Images " << endl << "| 3: Some Images | 4: Read image from file | :";
	cin >> n;
	if (n == 0)
		image_version = true;
	if (n == 1)
		image_version = false;
	if (n == 2 || n == 3)
		all = true;

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
				if (!(iss >> a )) { break; } // error

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

				//Cheat
				myGui->FILE_NAME = n;

				cout << n << endl;
				vector<Mat> resultMats = myGui->ImageProcessing_Final(frame, false, true, true);
				imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\xx\\result_" + n + ".jpg", frame);
				for (int j = 0; j < resultMats.size(); j++) {
					imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\xx\\result_" + n + "(1).jpg", resultMats[j]);
				}
			}
		}
		double total;
		for (int i = 0; i < percentageOverlapDatas.size(); i++)
		{
			total += percentageOverlapDatas[i];
		}
		cout << "Average percentage of Overlap: " << total / percentageOverlapDatas.size() << endl;
		return 0;
	}

	if (all) {
		int from = 2, to = 200;

		if (n == 3) {
			cout << "From: ";
			cin >> from;
			cout << "To: ";
			cin >> to;
		}

		for (int i = from; i <= to; i++) {
			string n = to_string(i);
			for (int i = n.size(); i < 3; i++) {
				n = "0" + n;
			}
			string path = pathData + n + ".jpg";

			

			frame = cv::imread(path, CV_LOAD_IMAGE_COLOR);
			if (!frame.data) {
				continue;
			}
			pathUserInput = pathUserInput + n + ".txt";

			MYcppGui *myGui = new MYcppGui();

			myGui->AddUserInput(pathUserInput);
			
			//Cheat
			myGui->FILE_NAME = n;
			cout << n << endl;
			try {
				vector<Mat> resultMats = myGui->ImageProcessing_Final(frame, false, true, true);
				//string t = GetTime();
				imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\2\\result_" + n + ".jpg", frame);
				for (int j = 0; j < resultMats.size(); j++) {
					imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\2\\result_" + n + "(1).jpg", resultMats[j]);
				}

				//
				
			}
			catch (std::out_of_range& exc) {
				std::cerr << exc.what();
				cout << "Error!" << endl;
				continue;
			}
			catch (...){
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

		//Correct form input
		for (int i = fileName.size(); i < 3; i++) {
			fileName = "0" + fileName;
		}

		pathUserInput = pathUserInput + fileName + ".txt";
		MYcppGui *myGui = new MYcppGui();
		myGui->AddUserInput(pathUserInput);

		//for image
		if (image_version) {
			pathData = pathData + fileName + ".jpg";

			//Cheat
			myGui->FILE_NAME = fileName;
			frame = cv::imread(pathData, CV_LOAD_IMAGE_COLOR);
			myGui->ImageProcessing_Final(frame, false, true, true);
		}
		//for video version
		else {
			pathData = pathData + "video_" + fileName + ".avi";
			myGui->VideoProcessing(pathData);
		}
		waitKey(0);
	}
	return 0;
}