// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <thread>
#include "stdafx.h"
#include "common.h"

#define STANDARD_WIDTH 64
#define STANDARD_HEIGHT 128
#define HISTOGRAM_SIZE 7980
#define INPUT_WIDTH 100
#define INPUT_HEIGHT 200
#define POINT1 100
#define POINT2 100
#define MAX_QUEUES 40

struct data {
	std::vector<float> hog;
	int letterClass;
	float distance;

	bool operator<(const data& rhs) const {
		return distance > rhs.distance;
	}
};

std::vector<std::priority_queue<data>> qs;

std::priority_queue<data> q;

std::mutex g_push_mutex;

void initialize_queues() {

	for (int i = 0; i < MAX_QUEUES; i++) {
		qs.push_back(q);
	}
}

Mat preprocess_image(Mat image)
{
	Mat processed_image = Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);

	int height = image.rows;
	int width = image.cols;

	int delay = 0;

	for (int i = POINT1; i < 200 + POINT1; i++)
	{
		for (int j = POINT2; j < 100 + POINT2; j++)
		{
			Vec3b v3 = image.at<Vec3b>(i, j);
			processed_image.at<Vec3b>(i-POINT1, j-POINT2) = v3;
		}
	}

	resize(processed_image, processed_image, Size(STANDARD_WIDTH, STANDARD_HEIGHT));

	return processed_image;
}

std::vector<float> computeHOG(Mat image) {

	Mat gradientX, gradientY;
	Mat magnitude, angle_f;

	image.convertTo(image, CV_32F, 1.0 / 255.0);
	//compute gradients 
	Sobel(image, gradientX, CV_32F, 1, 0, 1);
	Sobel(image, gradientY, CV_32F, 0, 1, 1);

	//compute magnitude and direction of gradients
	
	cartToPolar(gradientX, gradientY, magnitude, angle_f, 1);

	int height = magnitude.rows;
	int width = magnitude.cols;
	float histogram[128][19];

	for (int j = 0; j < 128; j++) {
		for (int i = 0; i < 19; i++) {
			histogram[j][i] = 0.0;
		}
	}

	Mat angle = Mat(height, width, CV_32SC1);

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			angle.at<int>(row, col) = round( angle_f.at<float>(row, col));
		}
	}

	float magnitudeValue;
	int angleValue;
	int auxiliaryAngle;
	int bin1;
	int bin2;
	float bin1Value;
	float bin2Value;
	int squareIndex = 0;
	int rowStart;
	int colStart;

	for (int row_it = 0; row_it < height / 8; row_it++) {

		rowStart = row_it * 8;

		for (int col_it = 0; col_it < width / 8; col_it++) {

			colStart = col_it * 8;

			for (int row = rowStart; row < 8 + rowStart; row++) {
				for (int col = colStart; col < 8 + colStart; col++) {

					magnitudeValue = magnitude.at<float>(row, col);
					angleValue = angle.at<int>(row, col);

					if ((angleValue % 20 == 0)) {
						histogram[squareIndex][angleValue / 20] += magnitudeValue;
					}
					else {

						//vedem intre care doua binuri se regaseste unghiul
						auxiliaryAngle = angleValue % 20;
						bin1 = floor(angleValue / 20);
						bin2 = bin1 + 1;
						bin1Value = ((float)auxiliaryAngle / 20) * magnitudeValue;
						bin2Value = (1 - (float)auxiliaryAngle / 20) * magnitudeValue;

						histogram[squareIndex][bin1] += bin1Value;
						histogram[squareIndex][bin2] += bin2Value;

					}

				}
			
			}

			squareIndex++;

		}
	}

	//normalizare

	std::vector<float> normalizedHistogram;
	std::vector<float> vec1, vec2, vec3, vec4;
	int index1, index2, index3, index4;
	squareIndex = 0;

	for (int row = 0; row < height / 8 - 1; row++) {
		for (int col = 0; col < width / 8 - 1; col++) {

			index1 = row * 8 + col;
			index3 = row * 8 + (col + 1);
			index2 = (row + 1) * 8 + col;
			index4 = (row + 1) * 8 + (col + 1);

			vec1.clear();
			vec2.clear();
			vec3.clear();
			vec4.clear();

			for (int i = 0; i < 19; i++) {
				
				vec1.push_back(histogram[index1][i]);
				vec2.push_back(histogram[index2][i]);
				vec3.push_back(histogram[index3][i]);
				vec4.push_back(histogram[index4][i]);
			}

			vec1.insert(vec1.end(), vec2.begin(), vec2.end());
			vec3.insert(vec3.end(), vec4.begin(), vec4.end());
			vec1.insert(vec1.end(), vec3.begin(), vec3.end());

			normalize(vec1, vec1);
			normalizedHistogram.insert(normalizedHistogram.end(), vec1.begin(), vec1.end());

		}
	}

	imshow("magnitude", magnitude);
	imshow("gx", gradientX);
	imshow("gy", gradientY);

	return normalizedHistogram;

}

float computeDistance(std::vector<float> vector1, std::vector<float> vector2)
{
	float sum = 0;
	float distance;

	for (int i = 0; i < vector1.size(); i++) {
		sum += pow((vector1.at(i) - vector2.at(i)), 2);
	}

	distance = sqrt(sum);
	return distance;
}

void pushOntoQueue(std::vector<float> input, std::vector<data> database, unsigned int beginIndex, unsigned int endIndex, int threadIndex)
{
	
	for (unsigned int i = beginIndex; i < endIndex; i++)
	{
		database.at(i).distance = computeDistance(input, database.at(i).hog);
		g_push_mutex.lock();
		q.push(database.at(i));
		g_push_mutex.unlock();
	}
}

int classify(std::vector<float> input, std::vector<data> database, int k) 
{

	int num_threads = k;
	std::thread threads[100];
	unsigned int step = database.size() / num_threads;

	for (int i = 0; i < num_threads; i++) {
		threads[i] = (std::thread(pushOntoQueue, input, database,
			i * step, (i + 1) * step, i));
	}

	for (std::thread& t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}
	
	int frequency[25];
	for (int i = 0; i < 25; i++) {
		frequency[i] = 0;
	}

	for (int i = 0; i < k; i++) {
		frequency[q.top().letterClass]++;
		q.pop();
	}

	int max = 0;
	int max_i;
	for (int i = 0; i < 25; i++) {

		if (frequency[i] > max) {
			max = frequency[i];
			max_i = i;
		}
	}

	while (!q.empty()) {
		q.pop();
	}

	return max_i;
}

std::vector<data> readDatabase(const char* file, int letterClass) {

	printf("\nreading data... ");

	std::vector<data> db;
	FILE* f = fopen(file, "r");
	float value;

	int j = 0;

	while (!feof(f)) {

		std::vector<float> hog;

		if (!feof(f)) {

			for (int i = 0; i < HISTOGRAM_SIZE; i++) {


				fscanf(f, "%f", &value);
				hog.push_back(value);

				j++;
			}

			fscanf(f, "\n");

			data newData;
			newData.hog = hog;
			newData.distance = 0.0;
			newData.letterClass = letterClass;
			db.push_back(newData);

		}
	}

	printf("%d\n", db.size());
	return db;

}

void updateFile(const char* file, std::vector<float>data) {
	FILE* f = fopen(file, "a");

	for (int i = 0; i < data.size(); i++) {
		fprintf(f, "%f ", data.at(i));
	}

	fprintf(f, "\n");

	fclose(f);
}

void printResult(int value) {

	switch (value) {
	case 1: printf("A "); break;
	case 2: printf("B "); break;
	case 3: printf("C "); break;
	case 4: printf("D "); break;
	case 5: printf("E "); break;
	case 6: printf("F "); break;
	case 7: printf("G "); break;
	case 8: printf("H "); break;
	case 9: printf("I "); break;
	case 10: printf("K "); break;
	case 11: printf("L "); break;
	case 12: printf("M "); break;
	case 13: printf("N "); break;
	case 14: printf("O "); break;
	case 15: printf("P "); break;
	case 16: printf("Q "); break;
	case 17: printf("R "); break;
	case 18: printf("S "); break;
	case 19: printf("T "); break;
	case 20: printf("U "); break;
	case 21: printf("V "); break;
	case 22: printf("W "); break;
	case 23: printf("X "); break;
	case 24: printf("Y "); break;
	default: break;
	}

}

void start() {

	const char* file_1 = "histograms/A_HOG.txt";
	const char* file_2 = "histograms/B_HOG.txt";
	const char* file_3 = "histograms/C_HOG.txt";
	const char* file_4 = "histograms/D_HOG.txt";
	const char* file_5 = "histograms/E_HOG.txt";
	const char* file_6 = "histograms/F_HOG.txt";
	const char* file_7 = "histograms/G_HOG.txt";
	const char* file_8 = "histograms/H_HOG.txt";
	const char* file_9 = "histograms/I_HOG.txt";
	const char* file_10 = "histograms/K_HOG.txt";
	const char* file_11 = "histograms/L_HOG.txt";
	const char* file_12 = "histograms/M_HOG.txt";
	const char* file_13 = "histograms/N_HOG.txt";
	const char* file_14 = "histograms/O_HOG.txt";
	const char* file_15 = "histograms/P_HOG.txt";
	const char* file_16 = "histograms/Q_HOG.txt";
	const char* file_17 = "histograms/R_HOG.txt";
	const char* file_18 = "histograms/S_HOG.txt";
	const char* file_19 = "histograms/T_HOG.txt";
	const char* file_20 = "histograms/U_HOG.txt";
	const char* file_21 = "histograms/V_HOG.txt";
	const char* file_22 = "histograms/W_HOG.txt";
	const char* file_23 = "histograms/X_HOG.txt";
	const char* file_24 = "histograms/Y_HOG.txt";

	std::vector<data> database_A = readDatabase(file_1, 1);
	std::vector<data> database_B = readDatabase(file_2, 2);
	std::vector<data> database_C = readDatabase(file_3, 3);
	std::vector<data> database_D = readDatabase(file_4, 4);
	std::vector<data> database_E = readDatabase(file_5, 5);
	std::vector<data> database_F = readDatabase(file_6, 6);
	std::vector<data> database_G = readDatabase(file_7, 7);
	std::vector<data> database_H = readDatabase(file_8, 8);
	std::vector<data> database_I = readDatabase(file_9, 9);
	std::vector<data> database_K = readDatabase(file_10, 10);
	std::vector<data> database_L = readDatabase(file_11, 11);
	std::vector<data> database_M = readDatabase(file_12, 12);
	std::vector<data> database_N = readDatabase(file_13, 13);
	std::vector<data> database_O = readDatabase(file_14, 14);
	std::vector<data> database_P = readDatabase(file_15, 15);
	std::vector<data> database_Q = readDatabase(file_16, 16);
	std::vector<data> database_R = readDatabase(file_17, 17);
	std::vector<data> database_S = readDatabase(file_18, 18);
	std::vector<data> database_T = readDatabase(file_19, 19);
	std::vector<data> database_U = readDatabase(file_20, 20);
	std::vector<data> database_V = readDatabase(file_21, 21);
	std::vector<data> database_W = readDatabase(file_22, 22);
	std::vector<data> database_X = readDatabase(file_23, 23);
	std::vector<data> database_Y = readDatabase(file_24, 24);

	database_A.insert(database_A.end(), database_B.begin(), database_B.end());
	database_A.insert(database_A.end(), database_C.begin(), database_C.end());
	database_A.insert(database_A.end(), database_D.begin(), database_D.end());
	database_A.insert(database_A.end(), database_E.begin(), database_E.end());
	database_A.insert(database_A.end(), database_F.begin(), database_F.end());
	database_A.insert(database_A.end(), database_G.begin(), database_G.end());
	database_A.insert(database_A.end(), database_H.begin(), database_H.end());
	database_A.insert(database_A.end(), database_I.begin(), database_I.end());
	database_A.insert(database_A.end(), database_K.begin(), database_K.end());
	database_A.insert(database_A.end(), database_L.begin(), database_L.end());
	database_A.insert(database_A.end(), database_M.begin(), database_M.end());
	database_A.insert(database_A.end(), database_N.begin(), database_N.end());
	database_A.insert(database_A.end(), database_O.begin(), database_O.end());
	database_A.insert(database_A.end(), database_P.begin(), database_P.end());
	database_A.insert(database_A.end(), database_Q.begin(), database_Q.end());
	database_A.insert(database_A.end(), database_R.begin(), database_R.end());
	database_A.insert(database_A.end(), database_S.begin(), database_S.end());
	database_A.insert(database_A.end(), database_T.begin(), database_T.end());
	database_A.insert(database_A.end(), database_U.begin(), database_U.end());
	database_A.insert(database_A.end(), database_V.begin(), database_V.end());
	database_A.insert(database_A.end(), database_W.begin(), database_W.end());
	database_A.insert(database_A.end(), database_X.begin(), database_X.end());
	database_A.insert(database_A.end(), database_Y.begin(), database_Y.end());

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		printf("error opening the camera");
		return;
	}
	while (true)
	{
		Mat frame;
		cap.read(frame);
		cv::rectangle(frame, cv::Point(100,100), cv::Point(200, 300), cv::Scalar(0, 255, 0));

		Mat processed = preprocess_image(frame);
		std::vector<float> normalizedHistogram = computeHOG(processed);
		imshow("pre-processed image", processed);
		printResult(classify(normalizedHistogram, database_A, 15));

		flip(frame, frame, 1);

		imshow("camera", frame);
		if (waitKey(30) == 27)
		{
			return;
		}
	}
	
}

void createDatabase()
{

	Mat src1 = imread("used_images/Y/Y18.JPG");
	Mat src2 = imread("used_images/Y/Y19.JPG");
	Mat src3 = imread("used_images/Y/Y20.JPG");
	Mat src4 = imread("used_images/Y/Y21.JPG");
	Mat src5 = imread("used_images/Y/Y22.JPG");
	Mat src6 = imread("used_images/C/C6.JPG");
	Mat src7 = imread("used_images/C/C7.JPG");
	Mat src8 = imread("used_images/C/C8.JPG");
	Mat src9 = imread("used_images/C/C9.JPG");
	Mat src10 = imread("used_images/C/C10.JPG");
	
	Mat processed1 = preprocess_image(src1);

	imshow("image", processed1);
	waitKey(0);

	Mat processed2 = preprocess_image(src2);
	Mat processed3 = preprocess_image(src3);
	Mat processed4 = preprocess_image(src4);
	Mat processed5 = preprocess_image(src5);
	Mat processed6 = preprocess_image(src6);
	Mat processed7 = preprocess_image(src7);
	Mat processed8 = preprocess_image(src8);
	Mat processed9 = preprocess_image(src9);
	Mat processed10 = preprocess_image(src10);

	std::vector<float> normalizedHistogram1 = computeHOG(processed1);
	std::vector<float> normalizedHistogram2 = computeHOG(processed2);
	std::vector<float> normalizedHistogram3 = computeHOG(processed3);
	std::vector<float> normalizedHistogram4 = computeHOG(processed4);
	std::vector<float> normalizedHistogram5 = computeHOG(processed5);
	std::vector<float> normalizedHistogram6 = computeHOG(processed6);
	std::vector<float> normalizedHistogram7 = computeHOG(processed7);
	std::vector<float> normalizedHistogram8 = computeHOG(processed8);
	std::vector<float> normalizedHistogram9 = computeHOG(processed9);
	std::vector<float> normalizedHistogram10 = computeHOG(processed10);


	const char* file = "histograms/Y_HOG.txt";
	updateFile(file, normalizedHistogram1);
	updateFile(file, normalizedHistogram2);
	updateFile(file, normalizedHistogram3);
	updateFile(file, normalizedHistogram4);
	updateFile(file, normalizedHistogram5);
	updateFile(file, normalizedHistogram6);
	updateFile(file, normalizedHistogram7);
	updateFile(file, normalizedHistogram8);
	updateFile(file, normalizedHistogram9);
	updateFile(file, normalizedHistogram10);

}

int main()
{
	start();

	return 0;
}