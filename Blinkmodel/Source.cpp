#include <iostream>
#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Moments moment;
int gx, gy;

void main(int argc, char *argv[]){
	Mat matrix(Size(100, 100), CV_8UC1, Scalar::all(0));
	moment = moments(matrix, 1);
	gx = moment.m10 / moment.m00;
	gy = moment.m01 / moment.m00;
	cout << gx << endl;
	cout << gy << endl;
	circle(matrix, Point(gx, gy), 5, Scalar(255));
	namedWindow("rawCamera", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	imshow("rawCamera", matrix);
	while (1){
		int key = waitKey(10);
		if (key=='q'){
			destroyWindow("rawCamera");
			return;
		}
			
	}

}