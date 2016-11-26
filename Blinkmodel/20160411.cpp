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

//HoughCirle�p�p�����[�^
int const circleMinimumDistace = 100;						//HoughCirle�ϊ��Ŋ��m�����~�̍ŏ�����
int const circleMinimumRadious = 50;						//HoughCirle�ϊ��Ŋ��m�����~�̍ŏ����a
int const circleMaximumRadious = 200;						//HoughCirle�ϊ��Ŋ��m�����~�̍ő唼�a
double const houghCircleParamater1 = 100;					//HoughCircle�ϊ��̑��p�����[�^�[
double const houghCircleParameter2 = 50;					//HoughCircle�ϊ��̑��p�����[�^�[


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
			if (!cur) return;	//�J�[�\����������Ă��Ȃ��Ȃ甲����
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
	
	int key=0;		//key�͉����ꂽ�L�[
	int loopTime = 0;
	int frame = 0;										//�o�߃t���[��
	vector<PointerData> PointData(LightMax);					//vector�I�u�W�F�N�g�g����!!
	VideoCapture cap(0);
	cap.set( CV_CAP_PROP_FRAME_WIDTH, FrameWidth);
	cap.set( CV_CAP_PROP_FRAME_HEIGHT, FrameHeight);
	cap.set(CV_CAP_PROP_FPS, 30.0);
	cap.set( CV_CAP_PROP_CONVERT_RGB, false);
	//cap.set(CV_CAP_PROP_EXPOSURE, 0.1);

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

		cvtColor(rawCamera, Thresholded2, CV_BGR2GRAY);		//�O���C�X�P�[����
		threshold(Thresholded2, Thresholded2, GreyThreshold, 255, CV_THRESH_BINARY);
		
		//�t���[���̈ꕔ��{�����āA�V����LED����������B������10����

		for (int y = 0; y < FrameHeight; y += 50){
			const uchar *pLine = Thresholded2.ptr<uchar>(y);
			for (int x = 0; x < FrameWidth; x += 50){	
				if ( pLine[x] >GreyThreshold ){
					bool isNew = true;					//true if LED is found newly
					for (int pointNum = 0; pointNum < LightMax-1; pointNum++){
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
				//cout << endl;
			}
		}

		//cout << "----LED blink inspect----" << endl;

		//���ׂĂ�Point���d�S��p���Ĉʒu���̍X�V������
		
		double gx, gy;
		Moments moment;
		for (int pointNum = 0; pointNum < LightMax-1; pointNum++){		//check each LED on or off
			if (PointData[pointNum].getAlive()){							//�w�肵��point�������Ă���Ȃ�
					
				int cutposx = PointData[pointNum].getX() - LightMoveThreshold, cutposy = PointData[pointNum].getY() - LightMoveThreshold;	//cutpos�̎w����W����ʊO�ɂȂ��ăG���[�͂��̂�h��
				if (cutposx < 0) cutposx = 0;
				if (cutposx > FrameWidth - 2 * LightMoveThreshold-1) cutposx = FrameWidth - 2 * LightMoveThreshold-1;
				if (cutposy < 0) cutposy = 0;
				if (cutposy > FrameHeight - 2 * LightMoveThreshold-1) cutposy = FrameHeight - 2 * LightMoveThreshold-1;
				cout << "x=" << cutposx << ":y=" << cutposy << endl;
				

				Mat cut_img(Thresholded2, cvRect(cutposx, cutposy, 2 * LightMoveThreshold, 2 * LightMoveThreshold));				//LED���_���ӂ�؂���
				moment = moments(cut_img, 1);																						//�؂�����cut_img�ŐV�������[�����g���v�Z
				gx = moment.m10 / moment.m00;																						//�d�S��X���W
				gy = moment.m01 / moment.m00;																						//�d�S��y���W

				//cout << "check point[" << PointNum << "]:gx=" << gx << ":gy=" << gy;
				//cutposx = point[PointNum].getX() - LightMoveThreshold, cutposy = point[PointNum].getY() - LightMoveThreshold;
				//rectangle(Thresholded2, Point(cutposx, cutposy), Point(cutposx+2*LightMoveThreshold, cutposy+2*LightMoveThreshold), Scalar(255, 255, 255), 3, 4);

				if ((gx >= 0) && (gx <= 2 * LightMoveThreshold) && (gy >= 0) && (gy <= 2 * LightMoveThreshold)){					//gx,gy��cut_img�͈̔͂���͂ݏo���悤�Ȉُ�Ȓl���H
					int newX = PointData[pointNum].getX() + (int)(gx)-LightMoveThreshold;												//gx,gy�ɂ��X�V���ꂽXY���W��
					int newY = PointData[pointNum].getY() + (int)(gy)-LightMoveThreshold;
					bool isDub = false;																								//�X�V���ꂽ���W�͂ق���point�Ƃ��Ԃ��Ă��Ȃ����H
					for (int pointNumToCheckDublication = 0; pointNumToCheckDublication < pointNum; pointNumToCheckDublication++){
						if ((abs(PointData[pointNumToCheckDublication].getX() - newX)<LightMoveThreshold) && (abs(PointData[pointNumToCheckDublication].getY() - newY)<LightMoveThreshold)){
							isDub = true;
							break;
						}
					}

					if (!isDub){									//�ق���PointData�Əd�����Ȃ����Ƃ�����Ċm�F�ł�����
						PointData[pointNum].setXY(newX, newY);		//XY���W���X�V	
					}						
					//circle(Thresholded2, Point(point[PointNum].getX(), point[PointNum].getY()), 5, Scalar(0, 255, 255), -1, 8, 0);
					//cout << " :'0' " << endl;

				}else{

				}
				
				//cout << "(" << point[PointNum].getX() << "," << point[PointNum].getY() << "):"; 
		 		//cout << to_string(Thresholded2.at<uchar>(point[PointNum].getY(), point[PointNum].getX()));
				if (rawCamera.at<Vec3b>(PointData[pointNum].getY() | 1, PointData[pointNum].getX())[2] > RedThreshold){	//�d�S�_�̐ԐF������RedThreshold�ȏ�Ȃ�A��LED�_���ƌ���

					PointData[pointNum].addToBin(1);
					//std::cout << " :'1' :" << std::to_string(rawCamera.at<Vec3b>(point[PointNum].getY(), point[PointNum].getX())[2]) << endl;
				}else{
					PointData[pointNum].addToBin(0);
					//std::cout << " :'0' :" << std::to_string(rawCamera.at<Vec3b>(point[PointNum].getY(), point[PointNum].getX())[2]) <<  endl;
				}

				PointData[pointNum].drawLine();				//����`��
				PointData[pointNum].drawContourLine();		//Hough�ϊ��p�̗֊s����`��
				
			}else{

			}
		}

		//if (key == 'h'){
		if (frame % SpanHoughTransform==0){			//�n�t�ϊ����s��
			cout << "hough" << endl;
			//GaussianBlur(edgeImage, edgeImage, Size(9, 9), 2, 2);
			HoughCircles(edgeImage, circleData, CV_HOUGH_GRADIENT, 1, circleMaximumRadious, houghCircleParamater1, houghCircleParameter2, circleMinimumRadious, circleMaximumRadious);
			if (circleData.size() != 0){
				for (size_t i = 0; i < circleData.size(); i++){
					circle(disp, Point(circleData[i][0], circleData[i][1]), circleData[i][2], Scalar(100, 200, 0), 5, 8, 0);
				}
				edgeImage = Scalar(0, 0, 0);
			}
		}
		
		disp2 = disp.clone();	//disp2<-disp�R�s�[
		//rectangle(disp2, Point(DispFrameWidth, DispFrameHeight),Point(DispFrameWidth-320, DispFrameHeight-80), Scalar(255, 255, 255), CV_FILLED);

		
		for (int pointNum = 0; pointNum < LightMax-1; pointNum++){		//display bin to 
			/*
			putText(disp2, "ID=" + to_string(point[PointNum].getId()) + ":Color=", Point(DispFrameWidth - 300, DispFrameHeight - 25*LightMax + 25 * PointNum+25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));

			switch (point[PointNum].getColor())
			{
			case 0:		//�����S��
				putText(disp2,"T", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));
				break;
			case 1:		//��
				putText(disp2, "R", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 200));
				break;
			case 2:		//��
				putText(disp2, "B", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(200, 0, 0));
				break;
			case 3:		//��
				putText(disp2, "G", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 200, 0));
				break;
			default:
				putText(disp2, "U", Point(DispFrameWidth - 50, DispFrameHeight - 25 * LightMax + 25 * PointNum + 25), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));
				break;
			};

			//cout << "x=" + std::to_string(point[PointNum].getX()) + ":y=" + std::to_string(point[PointNum].getY()) + ":data=" + std::to_string(point[PointNum].getBin()) + ":l=" + std::to_string(point[PointNum].getLength()) + ":alive=" + std::to_string(point[PointNum].getAlive()) + ":work=" + std::to_string(point[PointNum].getWork()) <<  endl;
			//putText(Thresholded2, "(" + std::to_string(point[PointNum].getX()) + "," + std::to_string(point[PointNum].getY()) + "):id=" + std::to_string(point[PointNum].getId()) + ":color=" + std::to_string(point[PointNum].getColor()) + ":l=" + std::to_string(point[PointNum].getLength()) + ":cur=" + to_string(point[PointNum].getCur()) + ":debug =" + point[PointNum].debug() + "\n", Point(point[PointNum].getX(), point[PointNum].getY()), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
			*/
			putText(disp2, "id=" + to_string(PointData[pointNum].getId()) + ":color=" + to_string(PointData[pointNum].getColor()) + "BoR=" + to_string(rawCamera.at<Vec3b>(PointData[pointNum].getY() | 1, PointData[pointNum].getX())[2]), Point(DifDisplayX*PointData[pointNum].getX(), DifDisplayY*PointData[pointNum].getY()), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
			
			PointData[pointNum].drawCursor();
		}
		

		if (loopTime > 33){		//�t���[�����[�g�\��
			putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 35), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 200));
		}
		else{
			putText(disp2, "fps=" + to_string(loopTime), Point(DispFrameWidth - 80, DispFrameHeight - 35), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
		}
		putText(disp2, "GreyThreshold=" + to_string(GreyThreshold) + ":RedThreshold=" + to_string(RedThreshold) + ":BlueThreshold=" + to_string(BlueThreshold) + ":frame=" + to_string(frame) + "circleData.size()=" + to_string(circleData.size()), Point(0, DispFrameHeight - 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(200, 200, 200));
				
		cv::imshow("disp", disp2);
		cv::imshow("Thresholded2", Thresholded2);
		imshow("edgeImage", edgeImage);

		key = waitKey(1);

		if (key == 'q'){					//�I��
			destroyWindow("rawCamera");
			destroyWindow("Thresholded2");
			destroyWindow("edgeImage");
			return 0;
		}

		if (key == 'c'){
			disp = Scalar(0, 0, 0);			//disp��������
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

		loopTime = (cv::getTickCount() - time)*f;
		frame++;
	}	
}