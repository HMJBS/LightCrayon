#include <iostream>
#include <string>
#include <array>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int const FrameWidth = 720;					
int const FrameHeight = 480;
int const DispFrameWidth = 800;								//dispの横
int const DispFrameHeight = 600;							//dispの縦
int const GreyThreshold = 50;								//に値化のスレッショルド
int const LightSpaceThreshold = 100;						//光源かノイズかの閾値
int const LightMoveThreshold = 50;							//フレームごとに移動した光源の距離がこれより下ならば同一の光源と見る
int const LightMax = 3;										//光源数
int const BinDataLong = 5;									//バイナリデータのビット長
int const TtdLifetime = 60;									//Time to death　が何以上で削除
int const LineThickness = 10;								//線幅
int const RedThreshold = 100;								//赤色の閾値
int const BlueThreshold = 50;								//青色の閾値
float const DifDisplayX = DispFrameWidth / FrameWidth;		//xy座標にこれをかけるとdispでの座標になる
float const DifDisplayY = DispFrameHeight / FrameHeight;

Mat rawCamera;												//カメラ
Mat Thresholded2;											//rawcameraの表示用コピー
Mat disp(Size(DispFrameWidth, DispFrameHeight), CV_8UC3, Scalar(0, 0, 0));	//メイン画面　
Mat disp2(Size(DispFrameWidth, DispFrameHeight), CV_8UC3, Scalar(0, 0, 0));	//カーソルと合成されて実際に表示される方

class PointerData{

	private:

		int x, y, bin, l, buf,ttd;	//x axis, y axis , binary data, data length, LED創作ルーチンでLEDが発見され、binが更新されたか , buf:前回のbinを保存, ttd:消灯してからPointからさ駆除されるまでの時間
		bool alive, work;	//renewed:LED検索ルーチン内部で対応するLEDが発見されたか　alive:LEDがデータ転送中か work:画面内にLEDが存在するか
		int id;						//光クレヨンのid -1で未確定
		int color;					//カラーID　00:消しゴム　01:赤 10:青　11:緑
		bool cur;					//カーソル状態 true:押されている
		int lx, ly;					//直前のxy座標
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
			
			this->alive = false;
			this->work = false;
			this->id = -1;
			this->color = -1;
			this->cur = false;
			this->lx = 0;
			this->ly = 0;
		}
		void newPoint(int x, int y){
			this->x = x;
			this->y = y;
			this->lx = 0;
			this->ly = 0;
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
			
			this->alive = false;
			this->work = false;
			this->id = -1;
			this->color = -1;
			
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
		void addToBin(int dat){

			if (dat == 1){		//デバッグ情報
				debugdat.insert(0,"1");
			}
			else{
				debugdat.insert(0,"0");
			}

			if (rawCamera.at<Vec3b>(this->y, this->x)[0] > BlueThreshold){		//カーソル（青色）の検知
				this->cur = true;
			}
			else{
				this->cur = false;
			}

			if (dat == 0){
				ttd++;	//消灯ならばttdをインクリメント
			}
			else{
				ttd = 0;
			}

			if (work == false){
				if (dat == 1){
					work = true;		//aliveがfalseかつ入力が1ならば受付状態とする
				}
			}else{						//alive=trueつまり受付中
				bin = (bin << 1) + dat;
				l++;
				if (l > BinDataLong-1){
					work = false;
					buf = bin;
					bin = 0;
					l = 0;
					setIdColor();		
				}
			}
			
			
			if (ttd > TtdLifetime){		//ttdが式一以上ならば、LEDを殺す
				killPoint();
			}
			
			
		}
		void setXY(int x, int y){	
			this->lx = this->x;
			this->ly = this->y;
			this->x = x;
			this->y = y;
			if (this->x < 0) this->x = 0;
			if (this->x > FrameWidth-1) this->x = FrameWidth-1;		//x=720のピクセルは存在しない
			if (this->y < 0) this->y = 0;
			if (this->y > FrameHeight-1) this->y = FrameHeight-1;
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
		std::string debug(){
			return debugdat;
		}
		void setIdColor(){
			int parity = ((this->buf) & 1) + (((this->buf) & 2) >> 1) + (((this->buf) & 4) >> 2) + (((this->buf) & 8) >> 3) + (((this->buf) & 16) >> 4);
			if ((parity % 2) == 0){					//偶数のパリティビットのときのみ値を読み込む
				if (id == -1) id = (this->buf >> 3);	//上位2bitをidとする
				if (id == (this->buf >> 3)){
					this->color = (this->buf >> 1) & 3;			//下位2bitを色番号とする
				}
			}
		}
		int getId(){
			return this->id;
		}
		int getColor(){
			return this->color;
		}
		bool getCur(){
			return cur;
		}
		void drawLine(){
			if (!cur) return;	//カーソルが押されていないなら抜ける
			switch (this->color){
			case 1:
				line(disp, Point((int)(DifDisplayX*(float)this->lx), (int)(DifDisplayY*(float)this->ly)), Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)), Scalar(0, 0, 255), LineThickness, 4, 0);
				break;
			case 2:
				line(disp, Point((int)(DifDisplayX*(float)this->lx), (int)(DifDisplayY*(float)this->ly)), Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)), Scalar(255, 0, 0), LineThickness, 4, 0);
				break;
			case 3:
				line(disp, Point((int)(DifDisplayX*(float)this->lx), (int)(DifDisplayY*(float)this->ly)), Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)), Scalar(0, 255, 0), LineThickness, 4, 0);
				break;
			default:
				line(disp, Point((int)(DifDisplayX*(float)this->lx), (int)(DifDisplayY*(float)this->ly)), Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)), Scalar(0, 0, 0), LineThickness, 4, 0);
				break;
			}
		}
		void drawCursor(){
			circle(disp2, Point(DifDisplayX*this->x, DifDisplayY*this->y), LineThickness + 1, Scalar(255, 255, 255), 2, 4, 0);
			if (!this->cur){
				
				switch (this->color){
				case 1:
					circle(disp2, Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)),LineThickness, Scalar(0, 0, 255), -1, 4, 0);
					break;
				case 2:
					circle(disp2, Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)), LineThickness, Scalar(255, 0, 0), -1, 4, 0);
					break;
				case 3:
					circle(disp2, Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)), LineThickness, Scalar(0, 255, 0), -1, 4, 0);
					break;
				default:
					circle(disp2, Point((int)(DifDisplayX*(float)this->x), (int)(DifDisplayY*(float)this->y)), LineThickness, Scalar(0, 0, 0), -1, 4, 0);
					break;

				}
			}
		}
};
// 数値を２進数文字列に変換
string to_binString(unsigned int val){
	if (!val)
		return std::string("0");
	std::string str;
	while (val != 0) {
		if ((val & 1) == 0)  // val は偶数か？
			str.insert(str.begin(), '0');  //  偶数の場合
		else
			str.insert(str.begin(), '1');  //  奇数の場合
		val >>= 1;
	}
	return str;
}

int main(int argc, char *argv[]){
	
	int key=0;		//keyは押されたキー,fpsは１ルーチンの時間;
	int loopTime = 0;
	PointerData point[LightMax];
	VideoCapture cap(0);
	cap.set( CV_CAP_PROP_FRAME_WIDTH, FrameWidth);
	cap.set( CV_CAP_PROP_FRAME_HEIGHT, FrameHeight);
	cap.set(CV_CAP_PROP_FPS, 30.0);
	cap.set( CV_CAP_PROP_CONVERT_RGB, false);

	cout << "Initializing\n";
	
	if (!cap.isOpened()) {
		std::cout << "failed to capture camera";
			return -1;
	}

	namedWindow( "disp", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	namedWindow( "Thresholded2", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);

	while (1){

		double f = 1000.0 / cv::getTickFrequency();		//measure time from heressssss
		int64 time = cv::getTickCount();
		
		while (!cap.grab());
		cap.retrieve(rawCamera);

		cvtColor(rawCamera, Thresholded2, CV_BGR2GRAY);		//グレイスケール化
		threshold(Thresholded2, Thresholded2, GreyThreshold, 255, CV_THRESH_BINARY);

		
		for (int y = 0; y < FrameHeight; y += 50){
			const uchar *pLine = Thresholded2.ptr<uchar>(y);
			for (int x = 0; x < FrameWidth; x += 50){	//フレームの一部を捜査して、新しいLEDを検索する。ただし10刻み
				if ( pLine[x] >GreyThreshold ){
					bool isNew = true;					//true if LED is found newly
					for (int PointNum = 0; PointNum < LightMax-1; PointNum++){
						if ((abs(point[PointNum].getX() - x) < 100) & (abs(point[PointNum].getY() - y) < 100) & (point[PointNum].getAlive())){
							isNew = false;
							break;
						}
					}

					if (isNew){
						for (int PointNum = 0; PointNum < LightMax - 1; PointNum++){
							if (!point[PointNum].getAlive()){
								point[PointNum].newPoint(x, y);		//generator new LED
								break;
							}
						}
					}
				}
				//cout << endl;
			}
		}

		//cout << "----LED blink inspect----" << endl;
		
		double gx, gy;
		Moments moment;
		for (int PointNum = 0; PointNum < LightMax-1; PointNum++){		//check each LED on or off
			if (point[PointNum].getAlive()){
				
				int cutposx = point[PointNum].getX() - LightMoveThreshold, cutposy = point[PointNum].getY() - LightMoveThreshold;	//cutposの指定座標が悪い位置に行くのを防ぐ
				if (cutposx < 0) cutposx = 0;
				if (cutposx > FrameWidth - 2 * LightMoveThreshold-1) cutposx = FrameWidth - 2 * LightMoveThreshold-1;
				if (cutposy < 0) cutposy = 0;
				if (cutposy > FrameHeight - 2 * LightMoveThreshold-1) cutposy = FrameHeight - 2 * LightMoveThreshold-1;

				Mat cut_img(Thresholded2, cvRect(cutposx, cutposy, 2 * LightMoveThreshold, 2 * LightMoveThreshold));  
				moment = moments(cut_img, 1);
				gx = moment.m10 / moment.m00;
				gy = moment.m01 / moment.m00;

				//cout << "check point[" << PointNum << "]:gx=" << gx << ":gy=" << gy;
				//cutposx = point[PointNum].getX() - LightMoveThreshold, cutposy = point[PointNum].getY() - LightMoveThreshold;
				//rectangle(Thresholded2, Point(cutposx, cutposy), Point(cutposx+2*LightMoveThreshold, cutposy+2*LightMoveThreshold), Scalar(255, 255, 255), 3, 4);

				if ((gx >= 0) && (gx <= 2 * LightMoveThreshold) && (gy >= 0) && (gy <= 2 * LightMoveThreshold)){

					point[PointNum].setXY(point[PointNum].getX() + ((int)(gx)-LightMoveThreshold), point[PointNum].getY() + (int)(gy) - LightMoveThreshold);					
					//circle(Thresholded2, Point(point[PointNum].getX(), point[PointNum].getY()), 5, Scalar(0, 255, 255), -1, 8, 0);
					//cout << " :'0' " << endl;

				}else{

				}
				
				//cout << "(" << point[PointNum].getX() << "," << point[PointNum].getY() << "):"; 
				//cout << to_string(Thresholded2.at<uchar>(point[PointNum].getY(), point[PointNum].getX()));
				if (rawCamera.at<Vec3b>(point[PointNum].getY() | 1, point[PointNum].getX())[2] > RedThreshold){	//重心点の赤色成分がRedThreshold以上なら、赤LED点灯と見る

					point[PointNum].addToBin(1);
					//std::cout << " :'1' :" << std::to_string(rawCamera.at<Vec3b>(point[PointNum].getY(), point[PointNum].getX())[2]) << endl;
				}else{
					point[PointNum].addToBin(0);
					//std::cout << " :'0' :" << std::to_string(rawCamera.at<Vec3b>(point[PointNum].getY(), point[PointNum].getX())[2]) <<  endl;
				}

				point[PointNum].drawLine();		//線を描く
				
			}else{

			}
		}
		
		disp2 = disp.clone();	//disp2<-dispコピー
		rectangle(disp2, Point(DispFrameWidth, DispFrameHeight),Point(DispFrameWidth-320, DispFrameHeight-80), Scalar(255, 255, 255), CV_FILLED);

		for (int PointNum = 0; PointNum < LightMax-1; PointNum++){		//display bin to 

			putText(disp2, "ID=" + to_string(point[PointNum].getId()) + ":Color=", Point(DispFrameWidth - 300, DispFrameHeight - 25*LightMax + 25 * PointNum+25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));
			switch (point[PointNum].getColor())
			{
			case 0:		//消しゴム
				putText(disp2,"T", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));
				break;
			case 1:		//赤
				putText(disp2, "R", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 200));
				break;
			case 2:		//青
				putText(disp2, "B", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(200, 0, 0));
				break;
			case 3:		//緑
				putText(disp2, "G", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 200, 0));
				break;
			default:
				putText(disp2, "U", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));
				break;
			};

			//cout << "x=" + std::to_string(point[PointNum].getX()) + ":y=" + std::to_string(point[PointNum].getY()) + ":data=" + std::to_string(point[PointNum].getBin()) + ":l=" + std::to_string(point[PointNum].getLength()) + ":alive=" + std::to_string(point[PointNum].getAlive()) + ":work=" + std::to_string(point[PointNum].getWork()) <<  endl;
			//putText(Thresholded2, "(" + std::to_string(point[PointNum].getX()) + "," + std::to_string(point[PointNum].getY()) + "):id=" + std::to_string(point[PointNum].getId()) + ":color=" + std::to_string(point[PointNum].getColor()) + ":l=" + std::to_string(point[PointNum].getLength()) + ":cur=" + to_string(point[PointNum].getCur()) + ":debug =" + point[PointNum].debug() + "\n", Point(point[PointNum].getX(), point[PointNum].getY()), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
			putText(disp2, "id=" + to_string(point[PointNum].getId()) + ":color=" + to_string(point[PointNum].getColor()), Point(DifDisplayX*point[PointNum].getX(), DifDisplayY*point[PointNum].getY()), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
			point[PointNum].drawCursor();
		}


		if (loopTime > 33){		//フレームレート表示
			putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 200));
		}
		else{
			putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
		}
				
		cv::imshow("disp", disp2);
		cv::imshow("Thresholded2", Thresholded2);
		key = waitKey(1);

		if (key == 'q'){
			destroyWindow("rawCamera");
			destroyWindow("Thresholded2");
			return 0;
		}

		if (key == 'c'){
			disp = Scalar(0, 0, 0);			//dispを初期化
		}

		loopTime = (cv::getTickCount() - time)*f;
	}	
}