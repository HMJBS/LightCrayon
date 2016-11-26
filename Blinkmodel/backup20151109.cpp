#include <iostream>
#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int const FrameWidth = 720;					
int const FrameHeight = 480;
int const GreyThreshold = 80;				//に値化のスレッショルド
int const LightSpaceThreshold = 100;			//光源かノイズかの閾値
int const LightMovethreshold = 10;		//フレームごとに移動した光源の距離がこれより下ならば同一の光源と見る
int const LightMax = 1;						//光源数
int const BinDataLong = 4;					//バイナリデータのビット長
int const TtdLifetime = 60;					//Time to death　が何以上で削除
					

Mat rawCamera, Thresholded2, Thresholded, rect, raw;

vector<vector<Point>> contours;
Rect approxRect;

class PointerData{

	private:

		int x, y, bin, l, buf,ttd;	//x axis, y axis , binary data, data length, LED創作ルーチンでLEDが発見され、binが更新されたか , buf:前回のbinを保存, ttd:消灯してからPointからさ駆除されるまでの時間
		bool renewed, alive, work;	//renewed:LED検索ルーチン内部で対応するLEDが発見されたか　alive:LEDがデータ転送中か work:画面内にLEDが存在するか
		string debugdat;
	public:
		PointerData(){
			this->x = 0;
			this->y = 0;
			this->bin = 0;
			this->l = 0;
			this->buf = 0;
			this->ttd = 0;
			this->debugdat="";
			this->renewed=false;
			this->alive = false;
			this->work = false;
		}
		void newPoint(int x, int y){
			this->x = x;
			this->y = y;
			this->alive = true;
		}

		void killPoint(){
			this->x = 0;
			this->y = 0;
			this->bin = 0;
			this->l = 0;
			this->buf = 0;
			this->ttd = 0;
			this->debugdat = "";
			this->renewed = false;
			this->alive = false;
			this->work = false;
		}

		int getX(){
			return x;
		}
		int getY(){;
			return y;
		}
		int getBin(){
			return buf;
		}
		int getLength(){
			return l;
		}
		bool getRenewed(){
			return renewed;
		}
		void addToBin(int dat){
			if (dat == 1){
				debugdat.insert(0,"1");
			}
			else{
				debugdat.insert(0,"0");
			}
			if (work == false){
				if (dat == 0){
					work = true;		//aliveがfalseかつ入力が1ならば受付状態とする
					renewed = true;
				}
			}else{						//alive=trueつまり受付中
				bin = (bin << 1) + dat;
				if (dat == 0){
					ttd++;	//消灯ならばttdをインクリメント
				}else{
					ttd = 0;
				}
				renewed = true;
				l++;
				if (l > BinDataLong-1){
					work = false;
					buf = bin;
					bin = 0;
					l = 0;
				}
			}
			
		}


		void WriteCoordinate(int x, int y){
			this->x = x;
			this->y = y;
		}


		bool InspectAlive(){
			return alive;
		}
		void setRenewed(){
			renewed = true;
		}
		void clearRenewed(){
			renewed = false;
		}
		bool getAlive(){
			return alive;
		}
		bool getWork(){
			return work;
		}
		int getTTD(){
			return ttd;
		}
		string debug(){
			return debugdat;
		}
};


int main(int argc, char *argv[]){
	int key=0;
	PointerData point[LightMax];
	VideoCapture cap(0);
	cap.set( CV_CAP_PROP_FRAME_WIDTH, FrameWidth);
	cap.set( CV_CAP_PROP_FRAME_HEIGHT, FrameHeight);

	if (!cap.isOpened()) {
		cout << "failed to capture camera";
			return -1;
	}

	//cout << "camera captured";

	namedWindow( "rawCamera", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	namedWindow( "Thresholded", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);

	while (1){

		double f = 1000.0 / cv::getTickFrequency();		//measure time from here
		int64 time = cv::getTickCount();

		cap >> raw;
		resize(raw, rawCamera, Size(), 0.5, 0.5, INTER_NEAREST);
		cout << "cap detected" << endl;
		cvtColor(rawCamera, Thresholded, CV_BGR2GRAY);		//グレイスケール化
		threshold(Thresholded, Thresholded, GreyThreshold, 255, CV_THRESH_BINARY);
		Thresholded2 = Thresholded.clone();
		Thresholded.copyTo(Thresholded2);
		findContours(Thresholded, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		int contournum = 0;
		bool newflag = false;			//flag that LED is new or not
		for (auto contour = contours.begin(); contour != contours.end(); contour++){		//抽出したcontourを調査7

			newflag = false;

			cout << "search contour[" << to_string(contournum) << "]:";						//輪郭四角形approxRectを算出
			approxRect = boundingRect(*(contour));

			cout << "Rect=" << to_string(approxRect.width*approxRect.height);

			//calc rectangle of contours
			if (approxRect.width*approxRect.height < LightSpaceThreshold) {				//specify noize or not 
				 cout << ":Noize" << endl;
				
			}else{

				for (int PointNum = 0; PointNum < LightMax; PointNum++){	//すべてのPoint[]で回して、値を更新する

					if ((((approxRect.x - point[PointNum].getX()) ^ 2 + (approxRect.y - point[PointNum].getY()) ^ 2) < (LightMovethreshold^2)) && (point[PointNum].getRenewed()==false) && (newflag==false)){	//if gap of coordinate is low than LightMoveThreshold
						rectangle(Thresholded2, Point(approxRect.x, approxRect.y), Point(approxRect.x + approxRect.width, approxRect.y + approxRect.height), 255, 5, 8);
						point[PointNum].WriteCoordinate(approxRect.x, approxRect.y);	//renew coordinate of point struct
						point[PointNum].addToBin(1);
						newflag = true;			//LED was renewaled
						cout << "renewed Point[" << to_string(PointNum) << "]" << endl;

					}else{
						//nothing
					}
				}

				bool flagpoint = false;
				if (!newflag){												//newLED
					cout << "New LED";
					for (int PointNum = 0; PointNum < LightMax; PointNum++){
						if (!point[PointNum].getAlive() && !flagpoint ){					//register new LED to empty Point[]
							point[PointNum].newPoint(approxRect.x, approxRect.y);
							point[PointNum].addToBin(1);
							flagpoint = true;
							cout << ":registered as Point[" + std::to_string(PointNum) + "]" << endl;
						}
					}
					if (!flagpoint){
						cout << "not registered, maybe out of Point number" << endl;
					}
				}
			}


			contournum++;	//for debug
			cout << endl;
			
			//putText(Thresholded2, "x=" + std::to_string(point.getX()) + ":y=" + std::to_string(point.getY()) + ":data=" + std::to_string(point.getBin()) + ":l=" + std::to_string(point.getLength()), Point(point.getX(), point.getY() + 20), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200,200,200));
		}

		for (int PointNum = 0; PointNum < LightMax; PointNum++){		//check un_renewed Point object and write 0 to bin
			if (point[PointNum].getRenewed()==true){
				point[PointNum].clearRenewed();
				cout << "point[" << std::to_string(PointNum) << "]: renewed correctly" << endl;
			}else{
				point[PointNum].addToBin(0);
				point[PointNum].clearRenewed();
				cout << "point[" << std::to_string(PointNum) << "]: not renewed tought it is  0" << endl;
			}
			if (point[PointNum].getTTD() > TtdLifetime){
				cout << "point[" + std::to_string(PointNum) + "]: deleted because ttd is full" << endl;
				point[PointNum].killPoint();												//delete Point data
			}
			putText(Thresholded2, "x=" + std::to_string(point[PointNum].getX()) + ":y=" + std::to_string(point[PointNum].getY()) + ":data=" + std::to_string(point[PointNum].getBin()) + ":l=" + std::to_string(point[PointNum].getLength()) + ":alive=" + std::to_string(point[PointNum].getAlive()) + ":work=" + std::to_string(point[PointNum].getWork()) + ":ttd=" + std::to_string(point[PointNum].getTTD()) + "debug=" + point[PointNum].debug() , Point(0, 20 * PointNum + 20), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
		}

		for (int PointNum = 0; PointNum < LightMax; PointNum++){		//display bin to 
			cout << "x=" + std::to_string(point[PointNum].getX()) + ":y=" + std::to_string(point[PointNum].getY()) + ":data=" + std::to_string(point[PointNum].getBin()) + ":l=" + std::to_string(point[PointNum].getLength()) + ":alive=" + std::to_string(point[PointNum].getAlive()) + ":work=" + std::to_string(point[PointNum].getWork()) <<  endl;
		}

		imshow("rawCamera", rawCamera);
		imshow("Thresholded", Thresholded2);
		key = waitKey(10);
		if (key == 'q'){
			destroyWindow("rawCamera");
			destroyWindow("Thresholded");
			return 0;
		}

		std::cout << (cv::getTickCount() - time)*f << " [ms]" << std::endl;
		cout << "----------------------" << endl;
	}	
}