#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

double const FrameWidth = 1280;								//�J��������̉摜��X
double const FrameHeight = 720;								//�J��������̉摜��Y
double const DispFrameWidth = 800;							//disp�̉�
double const DispFrameHeight = 600;							//disp�̏c
int GreyThreshold = 47;										//2�l���̃X���b�V�����h
int const LightSpaceThreshold = 100;						//�������m�C�Y����臒l
int const LightMoveThreshold = 70;							//�t���[�����ƂɈړ����������̋����������艺�Ȃ�Γ���̌����ƌ���
int const LightMax = 4;										//������
int const BinDataLong = 5;									//�o�C�i���f�[�^�̃r�b�g��
int const TtdLifetime = 60;									//Time to death�@�����ȏ�ō폜c
int const LineThickness = 10;								//����
int const ContourThickness = 1;								//Hough�ϊ��p�̗֊s���C���[�W�ɂ�������̑���
int RedThreshold = 80;										//�ԐF��臒l
int BlueThreshold = 50;										//�F��臒l
int SpanHoughTransform = 30;								//�j���ϊ����s���t���[���Ԋu
double const DifDisplayX = DispFrameWidth / FrameWidth;		//xy���W�ɂ�����������disp�ł̍��W�ɂȂ�
double const DifDisplayY = DispFrameHeight / FrameHeight;
int mode = 2;												//�v���O�����̃��[�h�@0:���G���� 1:�{�[��

//HoughCirle�p�p�����[�^
int const circleMinimumDistace = 100;						//HoughCirle�ϊ��Ŋ��m�����~�̍ŏ�����
int const circleMinimumRadious = 50;						//HoughCirle�ϊ��Ŋ��m�����~�̍ŏ����a
int const circleMaximumRadious = 200;						//HoughCirle�ϊ��Ŋ��m�����~�̍ő唼�a
double const houghCircleParamater1 = 100;					//HoughCircle�ϊ��̑��p�����[�^�[
double const houghCircleParameter2 = 50;					//HoughCircle�ϊ��̑��p�����[�^�[

//PONG�p�p�����[�^�[
int const ballMax = 1;										//�{�[����
vector<vector<int>> ballDat(ballMax,vector<int>(4,0));		//�{�[���f�[�^�̎��[	 ballx,bally,ballAccerationX,BallAccerationY
int ballRadious = 10;										//�{�[���̔��a
int const barMax = 10;											//�o�[���\����������̐�
vector<vector<vector<int>>> bar(LightMax,vector<vector<int>>(barMax,vector<int>(4,0)));			//�o�[���\������e�����̍��W
int barThickness = 5;										//�o�[�̑���
Scalar barColor(255, 255, 255);								//�o�[�̐F
bool enableEdgeBounce = true;								//��ʒp�ŋʂ̓o�E���h���邩

//�p�l���p�p�����[�^
int const panelStat = 3;									//�p�l�����Ƃ肦���Ԃ̑���
int const panelDefaultX = 10, panelDefaultY = 10;							//x,y���ꂼ��̃p�l����
vector<Scalar> panelColor = { Scalar(150, 0, 0), Scalar(0, 150, 0), Scalar(0, 0, 150) };

Mat rawCamera;												//�J����
Mat Thresholded2;											//rawcamera�̕\���p�R�s�[
Mat disp(Size(static_cast<int>(DispFrameWidth), static_cast<int>(DispFrameHeight)), CV_8UC3, Scalar(0, 0, 0));			//���C����ʁ@
Mat disp2(Size(DispFrameWidth, DispFrameHeight), CV_8UC3, Scalar(0, 0, 0));			//�J�[�\���ƍ�������Ď��ۂɕ\��������
Mat edgeImage(Size(DispFrameWidth, DispFrameHeight), CV_8UC1, Scalar(0));		//�G�b�W���o���ꂽ�C���[�W

vector<Vec3f> circleData;																//Hough�ϊ��ɂ��~�������[����


class PointerData{							//��ʂɕ\�������LED���_���Ǘ�����N���X

	private:

		int x, y, bin, l, buf,ttd;			//x axis, y axis , binary data, data length, LED�n�샋�[�`����LED����������Abin���X�V���ꂽ�� , buf:�O���bin��ۑ�, ttd:�������Ă���Point���炳�쏜�����܂ł̎���
		bool alive, work;					//alive:LED���f�[�^�]������ work:��ʓ���LED�����݂��邩�@0:���݂��Ȃ��@1:���݂��AID�����肳��Ă���@4-2:���݂��邪ID�͌��蒆�@
		int id;								//���N��������id -1�Ŗ��m��
		int color;							//�J���[ID�@00:�����S���@01:�� 10:�@11:��
		bool cur;							//�J�[�\����� true:������Ă���
		int lx, ly;							//���O��xy���W
		vector<int> decidedat;				//id�����肷�邽�߂�3��ID��ǂݍ��݁A���������Ƃ�@
		string debugdat;					//�ߋ���bin��~��
		int bar[barMax][2];			//�{�[���f�[�^�̎��[	 ballx,bally,ballAccerationX,BallAccerationY
		int barIndex;						//�����O�o�b�t�@�̃C���f�b�N�X
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
		void addToBin(int dat){					//bin�Ƀf�[�^1,0������

			if (dat == 1){		//�f�o�b�O���
				debugdat.insert(0,"1");
			}
			else{
				debugdat.insert(0,"0");
			}

			if (rawCamera.at<Vec3b>(this->y, this->x)[0] > BlueThreshold){		//�J�[�\���i�F�j�̌��m
				this->cur = true;
			}
			else{
				this->cur = false;
			}

			if (dat == 0){
				ttd++;	//�����Ȃ��ttd���C���N�������g
			}
			else{
				ttd = 0;
			}

			if (work == false){
				if (dat == 1){
					work = true;		//alive��false�����͂�1�Ȃ�Ύ�t��ԂƂ���
				}
			}else{						//alive=true�܂��t��
				bin = (bin << 1) + dat;
				l++;
				if (l > BinDataLong-1){			//�f�[�^��l���S�f�[�^�ł���BinDataLong�܂肷�ׂẴf�[�^����M���I�����Ƃ��̏���
					work = false;
					buf = bin;
					bin = 0;
					l = 0;
					setIdColor();		
				}
			}
			
			
			if (ttd > TtdLifetime){		//ttd������ȏ�Ȃ�΁ALED���E��
				killPoint();
			}
			
			
		}
		void setXY(int x, int y){	
			this->lx = this->x;
			this->ly = this->y;
			this->x = x;
			this->y = y;
			if (this->x < 0) this->x = 0;
			if (this->x > FrameWidth-1) this->x = FrameWidth-1;		//x=720�̃s�N�Z���͑��݂��Ȃ�
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
			if ((parity % 2) == 0){					//�����̃p���e�B�r�b�g�̂Ƃ��̂ݒl��ǂݍ���
				if (id == -1) id = (this->buf >> 3);	//���2bit��id�Ƃ���
				if (id == (this->buf >> 3)){
					this->color = (this->buf >> 1) & 3;			//����2bit��F�ԍ��Ƃ���
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
			if (!cur && work!=0) return;	//�J�[�\����������Ă��Ȃ��Ȃ甲����
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
		void drawContourLine(){		//hough�ϊ��p�̗֊s�C���[�W��\������
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
		void drawFadingBar(){				//���΂炭���Ə�����o�[��`�悷��
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
		void addPosToBar(){					//�o�[�̕\�����W��������
			if ((this->x != 0) && (this->y != 0)){
				bar[barIndex][0] = this->x;
				bar[barIndex][1] = this->y;
				barIndex = (barIndex + 1) % barMax;
			}
		}
};
class Panel{
private:
	int panelX, panelY;		//x,y���ꂼ��̕����̃p�l������
	int pStatNum;			//panelDat
	vector<vector<int>> pDat;				//�j�p�l���̏��
	vector<Scalar> pColor;	//panelDat�̊e��ԂɑΉ�����F
public:
	Panel(){				//�R���X�g���N�^ px,py��xy���ꂼ��̃p�l����
		panelX = panelDefaultX;
		panelY = panelDefaultY;
		pStatNum = panelStat;
		pDat = vector<vector<int>>(panelX, vector<int>(panelY, 0));
		pColor = panelColor;
	}
	Panel(int px,int py):panelX(px),panelY(py){				//�R���X�g���N�^ px,py��xy���ꂼ��̃p�l����
		pStatNum = panelStat;
		pDat = vector<vector<int>>(panelX, vector<int>(panelY, 0));
		pColor = panelColor;
	}
	void set(int px, int py, int stat){	//�p�l��(x,y)��stat��ݒ�
		pDat[px][py] = stat;
	}
	int get(int px, int py){			//�p�l��(x,y)���擾
		return pDat[px][py];
	}
	void drawPanel(Mat& target){
		int pXnum, pYnum;		//�`�悷��p�l����xy���ꂼ��̐�
		float pWidth, pHeight;	//�`�悷��p�l����xy���ꂼ��̕�
		pXnum = target.cols;
		pYnum = target.rows;
		pWidth = pXnum / this->panelX;
		pHeight = pYnum / this->panelY;
		for (int x = 0; x < pXnum; x++){		//�p�l����rectangle�ŕ`��
			for (int y = 0; y < pYnum; y++){					
				rectangle(target, cvRect(static_cast<int>(x*pWidth), static_cast<int>(y*pHeight), static_cast<int>(pWidth), static_cast<int>(pHeight)), pColor[pDat[x][y]], -1);
			}
		}
	}
};
// ���l���Q�i��������ɕϊ�
string to_binString(unsigned int val){
	if (!val)
		return std::string("0");
	std::string str;
	while (val != 0) {
		if ((val & 1) == 0)  // val �͋������H
			str.insert(str.begin(), '0');  //  �����̏ꍇ
		else
			str.insert(str.begin(), '1');  //  ��̏ꍇ
		val >>= 1;
	}
	return str;
}

int main(int argc, char *argv[]){

	int key = 0;		//key�͉����ꂽ�L�[
	int loopTime = 0;
	int frame = 0;										//�o�߃t���[��
	vector<PointerData> PointData(LightMax);					//vector�I�u�W�F�N�g�g����!!
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