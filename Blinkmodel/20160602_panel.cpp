#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const float  FrameWidth = 1280;								//カメラからの画像のX  IEEEカメラなら720*480
const float  FrameHeight = 720;								//カメラからの画像のY  USBカメラなら 1280*720
const float  DispFrameWidth = 800;							//dispの横
const float  DispFrameHeight = 600;							//dispの縦
int findLightSpan = 30;									//LED探索ルーチンを行うフレーム間隔
int GreyThreshold = 47;									//2値化のスレッショルド
int  LightSpaceThreshold = 100;							//光源かノイズかの閾値
int LightMoveThreshold = 70;							//フレームごとに移動した光源の距離がこれより下ならば同一の光源と見る
const int  LightMax = 4;										//光源数=使用人数
const int  BinDataLong = 5;									//バイナリデータのビット長
int  TtdLifetime = 60;									//Time to death　LEDが見つからなかったときに増えるttdがこれ以上のとき、LEDは消失したと考える
const int MaxAllowedIdMismatch = 5;							//IDによるエラー検知の最大回数　これ以上ならばPointをkillする
int ContourThickness = 1;								//輪郭線の太さ
int  LineThickness = 10;								//線幅
int RedThreshold = 80;									//赤色の閾値
int BlueThreshold = 50;									//青色の閾値
int SpanHoughTransform = 30;							//破風変換を行うフレーム間隔
const float  DifDisplayX = DispFrameWidth / FrameWidth;		// dispFrame/Frame カメラ画像からウインドウへの座標変換 xy座標にこれをかけるとdispでの座標になる
const float  DifDisplayY = DispFrameHeight / FrameHeight;
int mode = 2;					//プログラムのモード　0:お絵かき 1:ボール 2:陣取りゲーム

vector<Scalar> penColor = { Scalar(0, 0, 0), Scalar(150, 0, 0), Scalar(0, 150, 0), Scalar(0, 0, 150), Scalar(150, 150, 0) };  //ペンの色　ID=0は黒
vector<Scalar> idColor = { Scalar(0, 150, 150), Scalar(150, 0, 0), Scalar(0, 150, 0), Scalar(0, 0, 150), Scalar(150, 150, 0) };  //IDの色

//パネル用パラメータ
int const panelStat = LightMax+1;									//パネルがとりえる状態の総数 すべてのIDの数 + 黒（ID未対応）
int const panelDefaultX = 10, panelDefaultY = 10;							//x,yそれぞれのパネル数
float panelDetectionRate = 0.8f;								//パネルの中心からこの値の比率だけの範囲で点が検出されたときにpanelのpDatを行進する　0.5ならば パネル単体のx軸長さ　*0.25 -0.75の範囲

Mat rawCamera;												//カメラ
Mat Thresholded2;											//rawcameraの表示用コピー
Mat disp(Size(static_cast<int>(DispFrameWidth), static_cast<int>(DispFrameHeight)), CV_8UC3, Scalar(0, 0, 0));			//メイン画面　
Mat disp2(Size(DispFrameWidth, DispFrameHeight), CV_8UC3, Scalar(0, 0, 0));			//カーソルと合成されて実際に表示される方
Mat edgeImage(Size(DispFrameWidth, DispFrameHeight), CV_8UC1, Scalar(0));		//エッジ検出されたイメージ

//for Ball
random_device rndBallDic;					//ボールの方向生産用
int maxBallSpeed = 64;						//ボールの最高速度
int defaultBallR = 20;						//ボールのデフォルト半径
Scalar defaultBallCol = Scalar(0, 255, 0);	//ボールのデフォルト色赤
//for PlayerBar
const int lineNum = 10;							//ひとつのPlayerBarを構成する線数
const int lineThickness = 5;
const int PlayerBarRenewInterbal = 3;	
const int maxLineLength = 30;

//for pong
const int MaxBall = 1;
const int ShowPointX = 250;
const int ShowPointY = 80;
const int ShowPointScale = 3;
const int pointDispTime = 90;		//30hz 3s
int isPlaying = 0;
int pongCnt = 0;					//pong用カウンタ変数

class PointerData{							//画面に表示されるLED光点を管理するクラス

	private:

		int x, y, bin, l, buf,ttd;			//x axis, y axis , binary data, data length, LED創作ルーチンでLEDが発見され、binが更新されたか , buf:前回のbinを保存, ttd:消灯してからPointからさ駆除されるまでの時間
		int allowedIdMis;					//IDエラー検地によるエラーの許される回数
		bool alive, work;					//alive:LEDがデータ転送中か work:画面内にLEDが存在するか　0:存在しない　1:存在し、IDが決定されている　4-2:存在するがIDは決定中　
		int id;								//光クレヨンのid -1で未確定
		int color;							//カラーID　00:消しゴム　01:赤 10:青　11:緑
		bool cur;							//カーソル状態 true:押されている
		int lx, ly;							//直前のxy座標
		vector<int> decidedat;				//idを決定するために3回IDを読み込み、多数決をとる　
		string debugdat;					//過去のbinを蓄積
		int barIndex;						//リングバッファのインデックス
	public:
		PointerData(){
			this->x = 0;
			this->y = 0;
			this->bin = 0;
			this->l = 0;
			this->buf = 0;
			this->ttd = 0;
			this->allowedIdMis = 0;
			this->debugdat="";
			this->alive = false;
			this->work = false;
			this->id = -1;
			this->color = -1;
			this->cur = false;
			this->lx = 0;
			this->ly = 0;
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
			this->allowedIdMis = 0;
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
			if (allowedIdMis > MaxAllowedIdMismatch){	//IDによるエラーを指定回数連続で検知されたらPointを殺す
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
			if ((parity % 2) == 0){								//偶数のパリティビットのときのみ値を読み込む
				if (id == -1) id = (this->buf >> 3);			//上位2bitをidとする
				if (id == (this->buf >> 3)){					//読み取れたIDはPoint.idと等しいか？
					this->color = (this->buf >> 1) & 3;			//下位2bitを色番号とする
					this->allowedIdMis = 0;
				}
				else{
					allowedIdMis++;					//IDによるエラー検知
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
			//cout << "this->x=" << to_string(DifDisplayX*this->x) << ",this->y=" << to_string(DifDisplayY*this->y) << endl;
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
		void setCur(bool in){
			this->cur = in;
		}
		void setId(int in){
			this->id = in;
		}
};
class Panel{
private:
	int panelX, panelY;		//x,yそれぞれの方向のパネル枚数
	int pStatNum;			//panelDat
	int pDat[panelDefaultX][panelDefaultY];				//核パネルの状態
	vector<Scalar> pColor;	//panelDatの各状態に対応する色
public:
	Panel(){				//コンストラクタ px,pyはxyそれぞれのパネル数
		panelX = panelDefaultX;
		panelY = panelDefaultY;
		pStatNum = panelStat;
		for (int i=0; i < panelDefaultX - 1; i++){
			for (int j=0;j < panelDefaultY - 1; j++){
				pDat[i][j] = 0;
			}
		}
		pColor = penColor;
	}
/*			パネルのxy軸ごとの枚数を引数にしたコンストラクタを作りたいけど、int[][]の宣言の仕方がわからん　後回し
	Panel(int px,int py):panelX(px),panelY(py){				//コンストラクタ px,pyはxyそれぞれのパネル数
		pStatNum = panelStat;
		for (int i = 0; i < panelDefaultX - 1; i++){
			for (int j = 0; j < panelDefaultY - 1; j++){
				pDat[i][j] = 0;
			}
		}
		pColor = panelColor;
	}
*/
	void set(int px,int py, int stat){	//パネル座標(x,y)にstatを設定
		pDat[px][py] = stat+1;
	}
	void setByScrPos(int px, int py,int stat,const Mat& target){	//スクリーン上の座標で入力できるset(),描画したいディスプレイの大きさを取得するためにMatを指定
		float pXnum, pYnum;		//描画するパネルのxyそれぞれの数
		float pWidth, pHeight;	//描画するパネルのxyそれぞれの幅
		float pXnumOffset, pYnumOffset;	//パネルの原点からどれだけ離れているか
		pXnum = static_cast<float>(target.cols);
		pYnum = static_cast<float>(target.rows);
		pWidth = pXnum / this->panelX;
		pHeight = pYnum / this->panelY;
		pXnumOffset = px % static_cast<int>(pWidth);		//0 <= pXnumOffset <= panelXのはず 
		pYnumOffset = py % static_cast<int>(pHeight);
		//cout << "px/pWidth,py/PHeight)=( " << floor(px / pWidth) << "," << floor(py / pHeight) << ")(pXnumOffset,pYnumOffset)=(" << to_string(pXnumOffset) << "," << to_string(pYnumOffset) << ")" << endl;
		if ((pXnumOffset > panelDetectionRate / 2 * pWidth) && ((panelDetectionRate / 2 + 0.5)*pWidth > pXnumOffset) && (pYnumOffset > panelDetectionRate / 2 * pHeight) && (panelDetectionRate / 2 + 0.5)*pHeight > pYnumOffset){
			set(floor(px / pWidth*DifDisplayX), floor(py / pHeight*DifDisplayY), stat);
			
		}
	}
	int get(int px,int py){			//パネル(x,y)を取得
		return pDat[px][py];
	}
	void drawPanel(Mat& target){
		float pXnum, pYnum;		//描画するパネルのxyそれぞれの数
		float pWidth, pHeight;	//描画するパネルのxyそれぞれの幅
		int stat_buf;
		pXnum = static_cast<float>(target.cols);
		pYnum = static_cast<float>(target.rows);
		pWidth = pXnum / this->panelX;
		pHeight = pYnum / this->panelY;
		for (int x = 0; x < panelX-1; x++){		//パネルをrectangleで描画
			for (int y = 0; y < panelY-1; y++){	
				//cout << "x,y=(" << to_string(x) << "," << to_string(y) << ") stat=" << to_string(pDat[x][y]) << endl;
				stat_buf = pDat[x][y];
				if (stat_buf != 0){
					rectangle(target, cvRect(static_cast<int>(x*pWidth), static_cast<int>(y*pHeight), static_cast<int>(pWidth), static_cast<int>(pHeight)), pColor[stat_buf], -1);
				}
			}
		}
	}

};
class Ball{
private:
	float x,y,r;	//x座標,y座標,半径
	float ax, ay;	//x,y軸への移動量
	Scalar col;		//色
	int stat;		//1-なら有効
	
public:
	bool refR, refL, refU, refD;	//反射したなら1
	Ball(){
		x = static_cast<int>(DispFrameWidth / 2);
		y = static_cast<int>(DispFrameHeight / 2);
		r = defaultBallR;
		ax = 5;
		ay = 8;
		col = defaultBallCol;
		stat = 0;
		refR = false;
		refL = false;
		refU = false;
		refD = false;
	}
	void activate(){
		stat = 1;
	}
	void deactivate(){
		stat = 0;
	}
	void setDafault(){
		x = static_cast<int>(DispFrameWidth / 2);
		y = static_cast<int>(DispFrameHeight / 2);
		ax = 5;
		ay = 8;
	}
	void move(){		//ax,ayだけボールを動かす
		//cout << "ax:" << ax << "ay:" << ay << endl;
		refR = false;
		refL = false;
		refU = false;
		refD = false;
		int tx=x, ty=y;		//直前のxy
		x = x + ax;
		y = y + ay;
		if ((x - r < 0)){		//x軸の画面端衝突
			x = tx;
			ax = -ax;
			refL = true;
		} 
		if (x + r > DispFrameWidth){	//右衝突
			x = tx;
			ax = -ax;
			refR = true;
		}
		if ((y - r < 0)){		//y軸の画面短衝突
			y = ty;
			ay = -ay;
			refU = true;
		}
		if (y + r > DispFrameHeight){
			y = ty;
			ay = -ay;
			refD = true;
		}
	}
	void setAccel(float pax,float pay){
		ax = pax;
		ay = pay;
	}
	void setAccelX(float pax){
		ax = pax;
	}
	void setAccelY(float pay){
		ay = pay;
	}
	void setPos(int px, int py){
		x = px;
		y = py;
	}
	float getX(){
		return x;
	}
	float getY(){
		return y;
	}
	float getR(){
		return r;
	}
	float getAX(){
		return ax;
	}
	float getAY(){
		return ay;
	}

	void draw(Mat dest){
		if (stat == 1) {
			circle(dest, Point(x, y), r, col);
		}
	}
};
class PlayerBar{
private:
	int ringBufIndex;	//Barを格納する配列のリングバッファ的先頭の要素
	float barArray[lineNum+1][2];
	int colorIndex;		//バーの色　idColor参照
	int stat,point;			//1に動作
public:
	PlayerBar(){
		colorIndex = 0;
		for (int i = 0; i < lineNum + 1; i++){
			barArray[i][0] = 0;
			barArray[i][1] = 0;
		}
		point = 0;
	}
	PlayerBar(int colInd){
		ringBufIndex = 0;
		colorIndex = colInd;
		for (int i = 0; i < lineNum + 1; i++){
			barArray[i][0] = 0;
			barArray[i][1] = 0;
		}
	}
	void activate(){
		stat = 1;
	}
	void deactivate(){
		stat = 0;
	}
	void addBar(int px, int py){		//頂点を追加
		barArray[ringBufIndex][0] = DifDisplayX*px;
		barArray[ringBufIndex][1] = DifDisplayY*py;
		ringBufIndex = (ringBufIndex + 1) % (lineNum + 1);		//リングバッファを進める
	}
	void addSlowBar(int px, int py){		//頂点を追加

		int prvInd = (ringBufIndex - 1) % (lineNum + 1);
		int apx = barArray[prvInd][0], apy = barArray[prvInd][1];		//ひとつ前の頂点座標
		int s2;															//長さの二乗
		if (apx *apy != 0){		//前の座標が初期値ではないか
			s2 = ((px - apx) ^ 2 + (py - apy) ^ 2);
			if (s2 < maxLineLength^2){
				barArray[ringBufIndex][0] = px;
				barArray[ringBufIndex][1] = py;
			}
			else{		//maxLineLength以上の線を描かない
				barArray[ringBufIndex][0] = maxLineLength*(px-apx) / sqrt(s2) + apx;
				barArray[ringBufIndex][1] = maxLineLength*(py-apy) / sqrt(s2) + apy;			//古い点から新しい点へのベクトルを正規化して、maxLIneLengthの長さのベクトルを生成
			}
		}
		ringBufIndex = (ringBufIndex + 1) % (lineNum + 1);		//リングバッファを進める
	}
	void draw(Mat dest){
		if (stat = 1){
			for (int i = 0; i < lineNum + 0; i++){
				float ax = barArray[(i + ringBufIndex) % (lineNum + 1)][0], ay = barArray[(i + ringBufIndex) % (lineNum + 1)][1], bx = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][0], by = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][1];
				if (ax*ay*bx*by!=0) line(dest, Point(ax, ay), Point(bx,by ), idColor[colorIndex],LineThickness); //座標が0の頂点を持つ線は描かない
			}
		}
	}
	bool getCollideBallPos(Ball& obj,Point2f& poi,Point2f& newpos){		//ballと衝突しているか,衝突していたら反射ベクトルを返す http://marupeke296.com/COL_2D_No5_PolygonToCircle.html
		float ballx=obj.getX(),bally=obj.getY(), ballr=obj.getR();
		float ary1x, ary1y, ary2x, ary2y;
		float sx, sy, ax, ay,bx,by,absSA,absS,d,dotas,dotbs,sbr;			//d=|S×A|/|S| Sは終点-始点
		float nx, ny,nabs;													//衝突線の正規化法線ベクトル
		float fx=obj.getAX(), fy=obj.getAY();														//衝突時のベクトル
		float a;
		float rx, ry;
		bool ret=false;														//返り血
		for (int i = 0; i < lineNum + 0; i++){
			ary2x = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][0];
			ary2y = barArray[(i + ringBufIndex + 1) % (lineNum + 1)][1];
			ary1x = barArray[(i + ringBufIndex) % (lineNum + 1)][0];
			ary1y = barArray[(i + ringBufIndex) % (lineNum + 1)][1];
			if ((ary1x*ary2x == 0) || (ary2x*ary2y == 0)) continue;  //座標に(0,0)を含む場合計算に入れない
			sx = ary2x - ary1x;								//S =ary2-ary1
			sy = ary2y - ary1y;
			ax = ballx - ary1x;
			ay = bally - ary1y;
			absSA = abs(sx*ay-ax*sy);
			absS = sqrt(sx*sx + sy*sy);
			if (absS == 0) continue;
			d = absSA / absS;
			//cout << d << endl;
			if (d > ballr){
				continue;
			}
			else{
				bx = ballx - barArray[(i + ringBufIndex + 1 ) % (lineNum + 1)][0];
				by = bally - barArray[(i + ringBufIndex + 1 ) % (lineNum + 1)][1];
				if ((ax*sx + ay*sy)*(bx*sx + by*sy) > 0){
					continue;
				}
				else{			//衝突　反射ベクトルを    n=S×A=(sx,sy,0)×(sx,sy,1)=(sy,-sx,0)
					nabs = sqrt(sy*sy + sx*sx);			//nの絶対値
					nx = sy / nabs;
					ny = -sx / nabs;
					//r=f-2dot(f,n)*n
					//a=2(fxnx+fyny)
					a = 2*(fx*nx + fy*ny);
					rx = fx - a*nx;
					ry = fy - a*ny;
					if (rx*rx + ry*ry > maxBallSpeed*maxBallSpeed){
						float sum_r = sqrt(rx*rx + ry*ry);
						rx = rx / sum_r*maxBallSpeed;
						ry = ry / sum_r*maxBallSpeed;
					}
					sbr = sx*bally - sy*ballx;
					if (sbr < 0){
						newpos = Point2f((ballr - d)*nx, (ballr - d)*ny);
						cout << "+" << endl;
					}
					else{
						newpos = Point2f((ballr - d)*-nx, (ballr - d)*-ny);
						cout << "-" << endl;
					}
					poi=Point2f(fx - a*nx, fy - a*ny);
					
					cout << "d=" << d << ":from(" << fx << "," << fy << ") to (" << fx - a*nx << "," << fy - a*ny << ")" << endl;
					return true;
				}
			}
		}
		return false;
	}
	void addOneRingBuf(){			//リングバッファのみ勧める
		ringBufIndex = (ringBufIndex + 1) % (lineNum + 1);		//リングバッファを進める
	}
	void addPoint(int ip){
		point = point + ip;
	}
	int getPoint(){
		return point;
	}
	void setPoint(int ip){
		point = ip;
	}	
	void reset(){
		for (int i = 0; i < lineNum + 1; i++){
			barArray[i][0] = 0;
			barArray[i][1] = 0;
		}
	}
};

class Pong{
private:
	Ball b[MaxBall];
	PlayerBar p[LightMax];
	int stat;
public:
	Pong(){
		stat = 0;
	}
	void startGame(){	//ゲームを始める
		//ボールを生産
		for (int i = 0; i < MaxBall; i++){
			//			b[i] = Ball();
			b[i].activate();
		}
		//ゲームバーを生産
		for (int i = 0; i < LightMax; i++){
			p[i] = PlayerBar(i);
			p[i].activate();
		}
		stat = 1;		//ゲーム中
	}
	void endGame(){
		//ボールを非活性化
		for (int i = 0; i < MaxBall; i++){
			b[i].deactivate();
		}
		//ゲームバーを非活性化
		for (int i = 0; i < LightMax; i++){
			p[i].deactivate();
		}
		stat = 0;
	}
	void moveBalls(){								//ボールをすべて動かす
		if (stat == 1){
			for (int i = 0; i < MaxBall; i++){
				b[i].move();
			}
		}
	}
	void updateBars(vector<PointerData> source,bool isSlow){		//Point Source[LightMax]を元にすべてのプレイヤーバーを更新
		for (int i = 0; i < LightMax; i++){
			float srcx = source[i].getX(), srcy = source[i].getY();		//バーの新しい座標を取得
			if ((srcx != 0) || (srcy != 0)){							//座標が(0,0)でないか
				if (isSlow){
					p[i].addSlowBar(srcx, srcy);
				}
				else{
					p[i].addBar(srcx, srcy);
				}
			}
			else{
				p[i].addOneRingBuf();
			}
		}
	}

	void correctCollide(){					//すべてのバーでボールとの衝突を判定し、衝突していたら反射
		Point2f poi,newpos;
		for (int i = 0; i < LightMax; i++){
			for (int j = 0; j < MaxBall; j++){
				if (p[i].getCollideBallPos(b[i], poi,newpos)){
					b[j].setAccel(poi.x, poi.y);
					//b[j].setPos(b[j].getX() + newpos.x, b[j].getY() + newpos.y);
				}
				else{
				}
			}
		}
	}
	void draw(Mat dest){
		//ボールを描画
		for (int i = 0; i < MaxBall; i++){
			b[i].draw(dest);
		}
		//ゲームバーを描画
		for (int i = 0; i < LightMax; i++){
			p[i].draw(dest);
		}
	}
	void showPointVsStyle(int id1, int id2, Mat& dest){		//b[id1]とb[id2]の得点を表示
		putText(dest, to_string(p[id1].getPoint()) + " - " + to_string(p[id2].getPoint()), Point(ShowPointX, ShowPointY), FONT_HERSHEY_COMPLEX, ShowPointScale, Scalar(255, 255, 255));
	}
	void addBallScore(int id, int scr){
		p[id].addPoint(scr);
	}
	void setBallScore(int id, int scr){
		p[id].setPoint(scr);
	}
	void clearAllPoint(){
		for (int i = 0; i < LightMax; i++){
			p[i].setPoint(0);
		}
	}
	bool checkBallHitLeftWall(int ballid){
		return b[ballid].refL;
	}
	bool checkBallHitRightWall(int ballid){
		return b[ballid].refR;
	}
	void resetBallPos(int ballid){
		b[ballid].setDafault();
	}
	int getPlayerbarPoint(int idp){
		return p[idp].getPoint();
	}
	void activeBall(int ballid){
		b[ballid].activate();
	}
	void deactiveBall(int ballid){
		b[ballid].deactivate();
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
};

int main(int argc, char *argv[]){

		int key = 0;		//keyは押されたキー
		int loopTime = 0;
		int frame = 0;										//経過フレーム
		bool slowflag = false;									//スロープレイヤーバーのフラグ
		vector<PointerData> PointData(LightMax);					//vectorオブジェクト使えや!!
		VideoCapture cap(0);
		cap.set(CV_CAP_PROP_FRAME_WIDTH, FrameWidth);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, FrameHeight);

		Pong game;													//game
		game.startGame();

		cout << "Initializing\n";

		if (!cap.isOpened()) {
			std::cout << "failed to capture camera";
			return -1;
		}

		namedWindow("disp", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
		//namedWindow( "Thresholded2", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);


		while (1){

			//PointData[0].setXY(cx, cy);

			double f = 1000.0 / cv::getTickFrequency();		//measure time from heressssss
			int64 time = cv::getTickCount();

			while (!cap.grab());
			cap.retrieve(rawCamera);

			cvtColor(rawCamera, Thresholded2, CV_BGR2GRAY);		//グレイスケール化
			threshold(Thresholded2, Thresholded2, GreyThreshold, 255, CV_THRESH_BINARY);

			//フレームの一部を捜査して、新しいLEDを検索する。ただし10刻み
			if (frame%findLightSpan){
				for (int y = 0; y < FrameHeight; y += 50){
					const uchar *pLine = Thresholded2.ptr<uchar>(y);
					for (volatile int x = 0; x < FrameWidth; x += 50){
						if (pLine[x] > GreyThreshold){
							bool isNew = true;					//true if LED is found newly
							for (int pointNum = 0; pointNum < LightMax - 1; pointNum++){
								if ((abs(PointData[pointNum].getX() - x) < 100) & (abs(PointData[pointNum].getY() - y) < 100) & (PointData[pointNum].getAlive())){
									isNew = false;
									break;
								}
							}

							if (isNew){
								for (int PointNum = 0; PointNum < LightMax - 1; PointNum++){
									if (!PointData[PointNum].getAlive()){
										PointData[PointNum].newPoint(x, y);		//generator new LED
										break;
									}
								}
							}
						}
					}
				}
			}

			//すべてのPointを重心を用いて位置情報の更新をする

			double gx, gy;
			Moments moment;

			for (int pointNum = 0; pointNum < LightMax - 1; pointNum++){		//check each LED on or off
				if (PointData[pointNum].getAlive()){							//指定したpointが生きているなら

					int cutposx = PointData[pointNum].getX() - LightMoveThreshold, cutposy = PointData[pointNum].getY() - LightMoveThreshold;	//cutposの指定座標が画面外になってエラーはくのを防ぐ
					if (cutposx < 0) cutposx = 0;
					if (cutposx > FrameWidth - 2 * LightMoveThreshold - 1) cutposx = FrameWidth - 2 * LightMoveThreshold - 1;
					if (cutposy < 0) cutposy = 0;
					if (cutposy > FrameHeight - 2 * LightMoveThreshold - 1) cutposy = FrameHeight - 2 * LightMoveThreshold - 1;
					//cout << "x=" << cutposx << ":y=" << cutposy << endl;

					Mat cut_img(Thresholded2, cvRect(cutposx, cutposy, 2 * LightMoveThreshold, 2 * LightMoveThreshold));				//LED光点周辺を切り取る
					moment = moments(cut_img, 1);																						//切り取ったcut_imgで新しいモーメントを計算
					gx = moment.m10 / moment.m00;																						//重心のX座標
					gy = moment.m01 / moment.m00;																						//重心のy座標

					if ((gx >= 0) && (gx <= 2 * LightMoveThreshold) && (gy >= 0) && (gy <= 2 * LightMoveThreshold)){					//gx,gyがcut_imgの範囲からはみ出すような異常な値か？
						int newX = PointData[pointNum].getX() + (int)(gx)-LightMoveThreshold;												//gx,gyにより更新されたXY座標が
						int newY = PointData[pointNum].getY() + (int)(gy)-LightMoveThreshold;
						bool isDub = false;																								//更新された座標はほかのpointとかぶっていないか？
						for (int pointNumToCheckDublication = 0; pointNumToCheckDublication < pointNum; pointNumToCheckDublication++){
							if ((abs(PointData[pointNumToCheckDublication].getX() - newX) < LightMoveThreshold) && (abs(PointData[pointNumToCheckDublication].getY() - newY) < LightMoveThreshold)){
								isDub = true;
								break;
							}
						}

						if (!isDub){									//ほかのPointDataと重複がないことが晴れて確認できたら
							PointData[pointNum].setXY(newX, newY);		//XY座標を更新	
						}
					}
					else{

					}

					if (rawCamera.at<Vec3b>(PointData[pointNum].getY() | 1, PointData[pointNum].getX())[2] > RedThreshold){	//重心点の赤色成分がRedThreshold以上なら、赤LED点灯と見る

						PointData[pointNum].addToBin(1);
					}
					else{
						PointData[pointNum].addToBin(0);
					}
				}
				else{

				}
			}

			disp2 = disp.clone();	//disp2<-dispコピー

			//PONGの処理

			if (frame%PlayerBarRenewInterbal == 0){
				game.updateBars(PointData,false);
			}

			switch (isPlaying){			//pong中
			default:
				slowflag = false;
				game.deactiveBall(0);
				break;

			case 1:
				slowflag = true;
				game.activeBall(0);
				game.moveBalls();
				if (game.checkBallHitLeftWall(0)){		//左の壁にボールがぶつかったか
					game.addBallScore(0, 1);
					isPlaying = 2;
					pongCnt = pointDispTime;
					game.resetBallPos(0);
				}
				if (game.checkBallHitRightWall(0)){		//右の壁にボールがぶつかったか
					game.addBallScore(1, 1);
					isPlaying = 2;
					pongCnt = pointDispTime;
					game.resetBallPos(0);
				}
				break;

			case 2:							//isPlayingが1になるまで点数を表示
				slowflag = true;
				game.showPointVsStyle(0, 1, disp2);
				pongCnt--;
				if (pongCnt == 0) isPlaying = 1;
				if (game.getPlayerbarPoint(0) + game.getPlayerbarPoint(1) > 10){
					isPlaying = 3;
				}
				break;

			case 3:
				slowflag = true;
				game.showPointVsStyle(0, 1, disp2);
				break;
			}

			game.draw(disp2);
			if (frame > PlayerBarRenewInterbal*(lineNum + 1)) game.correctCollide();		//衝突判定
			for (int pointNum = 0; pointNum < LightMax - 1; pointNum++){		//display bin to 

				putText(disp2, "(x,y)=(" + to_string(PointData[pointNum].getX()) + "," + to_string(PointData[pointNum].getY()) + ")" + "id=" + to_string(PointData[pointNum].getId()) + ":color=" + to_string(PointData[pointNum].getColor()) + "BoR=" + to_string(rawCamera.at<Vec3b>(PointData[pointNum].getY() | 1, PointData[pointNum].getX())[2]), Point(DifDisplayX*PointData[pointNum].getX(), DifDisplayY*PointData[pointNum].getY()), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				PointData[pointNum].drawCursor();
			}

			if (loopTime > 33){		//フレームレート表示
				putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 35), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 200));
			}
			else{
				putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 35), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
			}

			game.draw(disp2);
			putText(disp2, "GreyThreshold=" + to_string(GreyThreshold) + ":RedThreshold=" + to_string(RedThreshold) + ":BlueThreshold=" + to_string(BlueThreshold) + ":frame=" + to_string(frame), Point(0, DispFrameHeight - 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));

			cv::imshow("disp", disp2);
			//cv::imshow("Thresholded2", rawCamera);
			//cv::imshow("edgeImage", edgeImage);

			key = waitKey(1);

			if (key == 'q'){					//終了
				destroyWindow("rawCamera");
				destroyWindow("Thresholded2");
				destroyWindow("edgeImage");
				return 0;
			}

			if (key == 'c'){
				disp = Scalar(0, 0, 0);			//dispを初期化
			}
			if (key == 'w'){					//greyThreshold +1
				if (GreyThreshold + 1 < 256) GreyThreshold++;
			}

			if (key == 's'){					//greyThreshold -1
				if (GreyThreshold - 1 >= 0) GreyThreshold--;
			}

			if (key == 'e'){					//RedThreshold +1
				if (RedThreshold + 1 < 256) RedThreshold++;
			}

			if (key == 'd'){					//RedThreshold -1
				if (RedThreshold - 1 >= 0) RedThreshold--;
			}
			if (key == 'r'){					//BlueThreshold +1
				if (BlueThreshold + 1 < 256) BlueThreshold++;
			}

			if (key == 'f'){					//blueThreshold -1
				if (BlueThreshold - 1 >= 0) BlueThreshold--;
			}
			if (key == 'i'){			//ゲーム開始
				isPlaying = 1;
				game.clearAllPoint();
			}
			if (key == 'o'){			//ゲーム終了
				isPlaying = 0;
				game.clearAllPoint();
			}

			loopTime = (cv::getTickCount() - time)*f;
			frame++;
		}
	}
