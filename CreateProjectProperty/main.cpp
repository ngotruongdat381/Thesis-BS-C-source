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

	cout << "0: Image | 1: Video : ";
	cin >> n;
	if (n == 0)
		image_version = true;
	if (n == 1)
		image_version = false;
	if (n == 2)
		all = true;

	

	if (all) {
		for (int i = 2; i < 100; i++) {
			string n = to_string(i);
			if (n.size() == 1) {
				n = "0" + n;
			}
			string path = pathData + n + ".jpg";
			frame = cv::imread(path, CV_LOAD_IMAGE_COLOR);
			if (!frame.data) {
				continue;
			}
			MYcppGui *myGui = new MYcppGui();
			myGui->AddUserInput(pathUserInput);
			cout << n << endl;
			try {
				vector<Mat> resultMats = myGui->ImageProcessing_Final(frame, false, true, true);
				string t = GetTime();
				imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\result_" + n + ".jpg", frame);
				for (int j = 0; j < resultMats.size(); j++) {
					imwrite("D:\\605\\Source code\\dataset\\Bulk Result\\result_" + n + "(1).jpg", resultMats[j]);
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
			fileName = "02";

		pathUserInput = pathUserInput + fileName + ".txt";
		MYcppGui *myGui = new MYcppGui();
		myGui->AddUserInput(pathUserInput);

		//for image
		if (image_version) {
			pathData = pathData + fileName + ".jpg";
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