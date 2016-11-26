//---------------------------------------------------------
// 概要     : CVテスト、グレイスケール、四角直接描画
//            ビット表示、2点追尾
// File Name : tsuikj2.cpp
// Library   : OpenCV 2.0
//---------------------------------------------------------

#include <iostream>
#include <array>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#define WIDTH	720	//	キャプチャ画像の横幅
#define HEIGHT	480	//	キャプチャ画像の縦幅
#define THRESH_BOTTOM	200		//	明度下限の閾値
#define THRESHOLD_MAX_VALUE	255	//	2値化の際に使用する最大値
#define CIRCLE_RADIUS	2		//	円の半径
#define LINE_THICKNESS	3		//	線の太さ
#define LINE_TYPE		8	//	線の種類
#define SHIFT			0	//	座標の小数点以下の桁を表すビット数

int main( int argc, char **argv ){ 
	int key;					//	キー入力用の変数
	CvCapture *capture = NULL;			//	カメラキャプチャ用の構造体
      	/* 画像を生成する*/ 
	IplImage *frameImage;			//	キャプチャ画像用IplImage
	IplImage *grayImage = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage
    IplImage *bwImage = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage
 //   IplImage *bw0Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage
   // IplImage *bw1Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage
//    IplImage *bw2Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage
 //   IplImage *bw3Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage
  //  IplImage *bw4Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage
   // IplImage *bw5Image = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );		//	グレースケール画像用IplImage

	char windowNameCapture[] = "Capture"; 
	//	キャプチャした画像を表示するウィンドウの名前

    
	//	カメラを初期化する
	if ( ( capture = cvCreateCameraCapture( 0 ) ) == NULL ) {
		//	カメラが見つからなかった場合
		printf( "カメラが見つかりません\n" );
		return -1;
	}

	//	ウィンドウを生成する
	cvNamedWindow( windowNameCapture, CV_WINDOW_AUTOSIZE );

	cvNamedWindow( "gray", CV_WINDOW_AUTOSIZE );

	cvNamedWindow( "bw", CV_WINDOW_AUTOSIZE );

  	//	初期背景を設定するためにカメラから画像を取得
	frameImage = cvQueryFrame( capture );
	printf("width %d, height %d \n", frameImage->width, frameImage->height);
	printf("nChannels %d, depth %d \n", frameImage->nChannels, frameImage->depth);

	//	メインループ
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
		//	captureの入力画像フレームをframeImageに格納する
		frameImage = cvQueryFrame( capture );
		//	frameImageをグレースケール化したものを、grayImageに格納する
		cvCvtColor( frameImage, grayImage, CV_BGR2GRAY );

		cvThreshold( grayImage, bwImage,100,200,CV_THRESH_BINARY);

//	cvCopy( bw0Image, bwImage, NULL);

		// 光点の検索
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
		// 未検出のとばし処理
		if( sw2 == 2){
			gx1_2 = gravityX;
			gy1_2 = gravityY;
			gx2_2 = gx2;
			gy2_2 = gy2;
		}

		// カーソル描画
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

		//	画像を表示する
		cvShowImage( windowNameCapture, frameImage );
	
		cvShowImage("gray",grayImage );
		cvShowImage("bw",bwImage );

		printf ("x= %3d ,y= %3d x= %3d ,y= %3d sw2= %d " , gx1_2, gy1_2, gx2_2, gy2_2, sw2);

		// デコード用のデータ抽出
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

		// デコード、4ビット
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
		//	キー入力判定
		key = cvWaitKey(1);				// 必ず必要。
		if( key == 'q' ) 
			//	'q'キーが押されたらループを抜ける
			break;
	
	}
	//	キャプチャを解放する
	cvReleaseCapture( &capture );
	//	メモリを解放する
	cvReleaseImage( &grayImage );

	//	ウィンドウを破棄する
	cvDestroyWindow( windowNameCapture );
	cvDestroyWindow( "gray" );

	return 0;
} 

