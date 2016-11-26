#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

double const FrameWidth = 1280;								//カメラからの画像のX
double const FrameHeight = 720;								//カメラからの画像のY
double const DispFrameWidth = 800;							//dispの横
double const DispFrameHeight = 600;							//dispの縦
int GreyThreshold = 47;										//2値化のスレッショルド
int const LightSpaceThreshold = 100;						//光源かノイズかの閾値
int const LightMoveThreshold = 70;							//フレームごとに移動した光源の距離がこれより下ならば同一の光源と見る
int const LightMax = 4;										//光源数
int const BinDataLong = 5;									//バイナリデータのビット長
int const TtdLifetime = 60;									//Time to death　が何以上で削除c
int const LineThickness = 10;								//線幅
int const ContourThickness = 1;								//Hough変換用の輪郭線イメージにおける線の太さ
int RedThreshold = 80;										//赤色の閾値
int BlueThreshold = 50;										//青色の閾値
int SpanHoughTransform = 30;								//破風変換を行うフレーム間隔
double const DifDisplayX = DispFrameWidth / FrameWidth;		//xy座標にこれをかけるとdispでの座標になる
double const DifDisplayY = DispFrameHeight / FrameHeight;
int mode = 2;												//プログラムのモード　0:お絵かき 1:ボール

//HoughCirle用パラメータ
int const circleMinimumDistace = 100;						//HoughCirle変換で感知される円の最小距離
int const circleMinimumRadious = 50;						//HoughCirle変換で感知される円の最小半径
int const circleMaximumRadious = 200;						//HoughCirle変換で感知される円の最大半径
double const houghCircleParamater1 = 100;					//HoughCircle変換の第一パラメーター
double const houghCircleParameter2 = 50;					//HoughCircle変換の第二パラメーター

//PONG用パラメーター
int const ballMax = 1;										//ボール数
vector<vector<int>> ballDat(ballMax,vector<int>(4,0));		//ボールデータの収納	 ballx,bally,ballAccerationX,BallAccerationY
int ballRadious = 10;										//ボールの半径
int const barMax = 10;											//バーを構成する線分の数
vector<vector<vector<int>>> bar(LightMax,vector<vector<int>>(barMax,vector<int>(4,0)));			//バーを構成する各線分の座標
int barThickness = 5;										//バーの太さ
Scalar barColor(255, 255, 255);								//バーの色
bool enableEdgeBounce = true;								//画面恥で玉はバウンドするか

//パネル用パラメータ
int const panelStat = 3;									//パネルがとりえる状態の総数
int const panelDefaultX = 10, panelDefaultY = 10;							//x,yそれぞれのパネル数
vector<Scalar> panelColor = { Scalar(150, 0, 0), Scalar(0, 150, 0), Scalar(0, 0, 150) };

Mat rawCamera;												//カメラ
Mat Thresholded2;											//rawcameraの表示用コピー
Mat disp(Size(static_cast<int>(DispFrameWidth), static_cast<int>(DispFrameHeight)), CV_8UC3, Scalar(0, 0, 0));			//メイン画面　
Mat disp2(Size(DispFrameWidth, DispFrameHeight), CV_8UC3, Scalar(0, 0, 0));			//カーソルと合成されて実際に表示される方
Mat edgeImage(Size(DispFrameWidth, DispFrameHeight), CV_8UC1, Scalar(0));		//エッジ検出されたイメージ

vector<Vec3f> circleData;																//Hough変換による円情報を収納する


class PointerData{							//画面に表示されるLED光点を管理するクラス

	private:

		int x, y, bin, l, buf,ttd;			//x axis, y axis , binary data, data length, LED創作ルーチンでLEDが発見され、binが更新されたか , buf:前回のbinを保存, ttd:消灯してからPointからさ駆除されるまでの時間
		bool alive, work;					//alive:LEDがデータ転送中か work:画面内にLEDが存在するか　0:存在しない　1:存在し、IDが決定されている　4-2:存在するがIDは決定中　
		int id;								//光クレヨンのid -1で未確定
		int color;							//カラーID　00:消しゴム　01:赤 10:青　11:緑
		bool cur;							//カーソル状態 true:押されている
		int lx, ly;							//直前のxy座標
		vector<int> decidedat;				//idを決定するために3回IDを読み込み、多数決をとる　
		string debugdat;					//過去のbinを蓄積
		int bar[barMax][2];			//ボールデータの収納	 ballx,bally,ballAccerationX,BallAccerationY
		int barIndex;						//リングバッファのインデックス
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
			this->bar;
			this->barIndex = 0;
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
			this -> decidedat.clear();
		
			
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
		void addToBin(int dat){					//binにデータ1,0を入れる

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
				if (l > BinDataLong-1){			//データ長lが全データであるBinDataLongつまりすべてのデータを受信し終えたときの処理
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
			if (!cur && work!=0) return;	//カーソルが押されていないなら抜ける
			switch (this->color){
			case 1:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(0, 0, 255), LineThickness, 4, 0);
				break;
			case 2:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(255, 0, 0), LineThickness, 4, 0);
				break;
			case 3:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(0, 255, 0), LineThickness, 4, 0);
				break;
			default:
				line(disp, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(0, 0, 0), LineThickness, 4, 0);
				break;
			}
		}
		void drawContourLine(){		//hough変換用の輪郭イメージを表示する
			if (!cur) return;
			line(edgeImage, Point(static_cast<int>(DifDisplayX*this->lx), static_cast<int>(DifDisplayY*this->ly)), Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), Scalar(255), ContourThickness, 4, 0);	
		}

		
		void drawCursor(){
			circle(disp2, Point(DifDisplayX*this->x, DifDisplayY*this->y), LineThickness + 1, Scalar(255, 255, 255), 2, 4, 0);
			if (!this->cur){
				
	
				switch (this->color){
				case 1:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(0, 0, 255), -1, 4, 0);
					break;
				case 2:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(255, 0, 0), -1, 4, 0);
					break;
				case 3:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(0, 255, 0), -1, 4, 0);
					break;
				default:
					circle(disp2, Point(static_cast<int>(DifDisplayX*this->x), static_cast<int>(DifDisplayY*this->y)), LineThickness, Scalar(0, 0, 0), -1, 4, 0);
					break;

				}
			}
		}
		void drawFadingBar(){				//しばらくたつと消えるバーを描画する
			int barX, barXX;
			for (volatile int j = 0; j < barMax-1; j++){
				
				if ((bar[j][0] != 0) && (bar[j][1] != 0) && (bar[j + 1][0] != 0) && (bar[j + 1][1] != 0)){
					barX = (j + barIndex) % barMax;
					barXX = (j + 1 + barIndex) % barMax;
					cout << to_string(barX) << endl;
					line(disp, Point(bar[barX][0], bar[barX][1]), Point(bar[barXX][0], bar[barXX][1]), barColor, barThickness);
				}
			}
		}
		void addPosToBar(){					//バーの表示座標を代入する
			if ((this->x != 0) && (this->y != 0)){
				bar[barIndex][0] = this->x;
				bar[barIndex][1] = this->y;
				barIndex = (barIndex + 1) % barMax;
			}
		}
};
class Panel{
private:
	int panelX, panelY;		//x,yそれぞれの方向のパネル枚数
	int pStatNum;			//panelDat
	vector<vector<int>> pDat;				//核パネルの状態
	vector<Scalar> pColor;	//panelDatの各状態に対応する色
public:
	Panel(){				//コンストラクタ px,pyはxyそれぞれのパネル数
		panelX = panelDefaultX;
		panelY = panelDefaultY;
		pStatNum = panelStat;
		pDat = vector<vector<int>>(panelX, vector<int>(panelY, 0));
		pColor = panelColor;
	}
	Panel(int px,int py):panelX(px),panelY(py){				//コンストラクタ px,pyはxyそれぞれのパネル数
		pStatNum = panelStat;
		pDat = vector<vector<int>>(panelX, vector<int>(panelY, 0));
		pColor = panelColor;
	}
	void set(int px, int py, int stat){	//パネル(x,y)にstatを設定
		pDat[px][py] = stat;
	}
	int get(int px, int py){			//パネル(x,y)を取得
		return pDat[px][py];
	}
	void drawPanel(Mat& target){
		int pXnum, pYnum;		//描画するパネルのxyそれぞれの数
		float pWidth, pHeight;	//描画するパネルのxyそれぞれの幅
		pXnum = target.cols;
		pYnum = target.rows;
		pWidth = pXnum / this->panelX;
		pHeight = pYnum / this->panelY;
		for (int x = 0; x < pXnum; x++){		//パネルをrectangleで描画
			for (int y = 0; y < pYnum; y++){					
				rectangle(target, cvRect(static_cast<int>(x*pWidth), static_cast<int>(y*pHeight), static_cast<int>(pWidth), static_cast<int>(pHeight)), pColor[pDat[x][y]], -1);
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

	int key = 0;		//keyは押されたキー
	int loopTime = 0;
	int frame = 0;										//経過フレーム
	vector<PointerData> PointData(LightMax);					//vectorオブジェクト使えや!!
	Panel firstPanel;
	Panel secondPanel(10, 10);

	
	VideoCapture cap(0);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, FrameWidth);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, FrameHeight);
	cap.set(CV_CAP_PROP_FPS, 30.0);
	cap.set(CV_CAP_PROP_CONVERT_RGB, false);

	while (1){
	}
}