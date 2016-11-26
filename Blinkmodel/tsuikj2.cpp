//---------------------------------------------------------
// �T�v     : CV�e�X�g�A�O���C�X�P�[���A�l�p���ڕ`��
//            �r�b�g�\���A2�_�ǔ�
// File Name : tsuikj2.cpp
// Library   : OpenCV 2.0
//---------------------------------------------------------

#include <iostream>
#include <array>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#define WIDTH	720	//	�L���v�`���摜�̉���
#define HEIGHT	480	//	�L���v�`���摜�̏c��
#define THRESH_BOTTOM	200		//	���x������臒l
#define THRESHOLD_MAX_VALUE	255	//	2�l���̍ۂɎg�p����ő�l
#define CIRCLE_RADIUS	2		//	�~�̔��a
#define LINE_THICKNESS	3		//	���̑���
#define LINE_TYPE		8	//	���̎��
#define SHIFT			0	//	���W�̏����_�ȉ��̌���\���r�b�g��

int main( int argc, char **argv ){ 
	int key;					//	�L�[���͗p�̕ϐ�
	CvCapture *capture = NULL;			//	�J�����L���v�`���p�̍\����
      	/* �摜�𐶐�����*/ 
	IplImage *frameImage;			//	�L���v�`���摜�pIplImage
	IplImage *grayImage = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage
    IplImage *bwImage = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage
 //   IplImage *bw0Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage
   // IplImage *bw1Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage
//    IplImage *bw2Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage
 //   IplImage *bw3Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage
  //  IplImage *bw4Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage
   // IplImage *bw5Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	�O���[�X�P�[���摜�pIplImage

	char windowNameCapture[] = "Capture"; 
	//	�L���v�`�������摜��\������E�B���h�E�̖��O

    
	//	�J����������������
	if ( ( capture = cvCreateCameraCapture( 0 ) ) == NULL ) {
		//	�J������������Ȃ������ꍇ
		printf( "�J������������܂���\n" );
		return -1;
	}

	//	�E�B���h�E�𐶐�����
	cvNamedWindow( windowNameCapture, CV_WINDOW_AUTOSIZE );

	cvNamedWindow( "gray", CV_WINDOW_AUTOSIZE );

	cvNamedWindow( "bw", CV_WINDOW_AUTOSIZE );

  	//	�����w�i��ݒ肷�邽�߂ɃJ��������摜���擾
	frameImage = cvQueryFrame( capture );
	printf("width %d, height %d \n", frameImage->width, frameImage->height);
	printf("nChannels %d, depth %d \n", frameImage->nChannels, frameImage->depth);

	//	���C�����[�v
	int x,y;	
	int n = 0;

	int gravityX;
	int gravityY;
	int gx2;
	int gy2;

/*	int gx1_0 = 60;
	int gy1_0 = 60;
	int gx2_0 = 60;
	int gy2_0 = 60;
*/
	int gx1_2;
	int gy1_2;
	int gx2_2;
	int gy2_2;

	unsigned char xx;
	unsigned char x2;

	unsigned char rdy;
	unsigned char rd2;

	int sw0 = 0;
	int s20 = 0;

	int nn;
	int n2;

	unsigned char xxx;
	unsigned char xx2;
	gx1_2 = gx2_2 = 60;
	gy1_2 = gy2_2 = 60;
	while( 1 ) {
		//	capture�̓��͉摜�t���[����frameImage�Ɋi�[����
		frameImage = cvQueryFrame( capture );
		//	frameImage���O���[�X�P�[�����������̂��AgrayImage�Ɋi�[����
		cvCvtColor( frameImage, grayImage, CV_BGR2GRAY );

		cvThreshold( grayImage, bwImage,100,200,CV_THRESH_BINARY);

//	cvCopy( bw0Image, bwImage, NULL);

		// ���_�̌���
		int sw2 = 0;
		gravityX = gx2 = 60;
		gravityY = gy2 = 60;
		for( y = 60;y< 480 ;y=y+10)
			for( x = 60;x < 720 ; x = x +10)
				if((unsigned char)( bwImage ->imageData[ bwImage -> widthStep * y + x ])){
					if( sw2 == 0){
						gravityX = x;
						gravityY = y;
						sw2 = 1;
					//	break;
					}
					else{
					if( sw2 == 1){
						if( (x < (gravityX - 50) ) || ((gravityX + 50) < x ) ||
							(y < (gravityY - 50) ) || ((gravityY + 50) < y )    ){
								gx2 = x;
								gy2 = y;
								sw2 = 2;
							}
					}
				}
		}

/*		if( gx1_0 != 80) 
		if( fabs((double)(gravityX - gx1_0) ) > 100)
			gravityX = gx1_0;

		if( gy1_0 != 80) 
		if( fabs((double)(gravityY - gy1_0) ) > 100)
			gravityY = gy1_0;

		gx1_0 = gravityX;
		gy1_0 = gravityY;
		gx2_0 = gx2;
		gy2_0 = gy2;
*/
		// �����o�̂Ƃ΂�����
		if( sw2 == 2){
			gx1_2 = gravityX;
			gy1_2 = gravityY;
			gx2_2 = gx2;
			gy2_2 = gy2;
		}

		// �J�[�\���`��
		for(x = -50 ;x < 50 ;x++){
//		for(x = 310 ;x < 410 ;x++){
			grayImage ->imageData[grayImage ->widthStep * (gy1_2 - 50 ) + x + gx1_2 ] = 200;
			grayImage ->imageData[grayImage ->widthStep * (gy2_2 - 50 ) + x + gx2_2 ] = 150; 
		}
		for( y = -50;y< 50;y++){
//		for( y = 190;y< 290;y++){
			grayImage ->imageData[grayImage ->widthStep * (y + gy1_2) + gx1_2 - 50 ] = 200; 
			grayImage ->imageData[grayImage ->widthStep * (y + gy2_2) + gx2_2 - 50 ] = 150; 
		}

//		cvCopy( bw3Image, bw4Image, NULL);
//		cvCopy( bw2Image, bw3Image, NULL);
//		cvCopy( bw1Image, bw2Image, NULL);	
//		cvCopy( bw0Image, bw1Image, NULL);
//		cvOr(bw0Image, bw1Image, bwImage, NULL);
//		cvOr(bw2Image, bw1Image, bwImage, NULL);

		//	�摜��\������
		cvShowImage( windowNameCapture, frameImage );
	
		cvShowImage("gray",grayImage );
		cvShowImage("bw",bwImage );

		printf ("x= %3d ,y= %3d x= %3d ,y= %3d sw2= %d " , gx1_2, gy1_2, gx2_2, gy2_2, sw2);

		// �f�R�[�h�p�̃f�[�^���o
		xx = 0;
		for( y = -24;y<= 24;y=y+2){
			for(x = -25 ;x <= 25 ;x++){
				if((unsigned char)( (grayImage ->imageData[
						grayImage -> widthStep * ((gy1_2 | 1)+ y) + gx1_2 +x ])) > 100)
					xx = 1; 
			} 
		}

		x2 = 0;
		for( y = -24;y<= 24;y=y+2){
			for(x = -25 ;x <= 25 ;x++){
				if((unsigned char)( (grayImage ->imageData[
						grayImage -> widthStep * ((gy2_2 | 1)+ y) + gx2_2 +x ])) > 100)
					x2 = 1; 
			} 
		}

		// �f�R�[�h�A4�r�b�g
		rdy = ' ';
		if(sw0 == 0){
			if( xx == 0 ){
				sw0 = 1;
				nn = 0;
				xxx = 0;
			}
		}
		else
		{	
	//		if(sw0 == 1)
	//		{
				xxx = xxx | (xx << nn);
				nn++;
				if(nn == 4){
					sw0 = 0;
					rdy = '*';
				
				}
				
	//		}
		}

		rd2 = ' ';
		if(s20 == 0){
			if( x2 == 0 ){
				s20 = 1;
				n2 = 0;
				xx2 = 0;
			}
		}
		else
		{	
	//		if(s20 == 1)
	//		{
				xx2 = xx2 | (x2 << n2);
				n2++;
				if(n2 == 4){
					s20 = 0;
					rd2 = '*';
				}
	//		}
		}

		printf("%2d %3d %3d %d %d %2d %c %d %d %2d %c\n",n % 30,
			(unsigned char)(grayImage ->imageData[grayImage ->widthStep * 241 + 360 ]),	
			(unsigned char)(grayImage ->imageData[grayImage ->widthStep * 240 + 360 ]),
			xx, sw0 , xxx, rdy, x2, s20 , xx2, rd2);

		n++;
		//	�L�[���͔���
		key = cvWaitKey(1);				// �K���K�v�B
		if( key == 'q' ) 
			//	'q'�L�[�������ꂽ�烋�[�v�𔲂���
			break;
	
	}
	//	�L���v�`�����������
	cvReleaseCapture( &capture );
	//	���������������
	cvReleaseImage( &grayImage );

	//	�E�B���h�E��j������
	cvDestroyWindow( windowNameCapture );
	cvDestroyWindow( "gray" );

	return 0;
} 

