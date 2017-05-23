#include "pch.h"

///////////////// My Data

//#define IMAGE_FILENAME "../../data/" "ball.png"


///////////////// SBU DATASET

/* good */
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd125.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd44.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd566.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd3548.jpg"


/* fair */
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd9.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd141.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd145.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd667.jpg"


/* bad */
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd281.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd15.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd408.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd4075.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd4102.jpg"
//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd1214.jpg"

//#define IMAGE_FILENAME "../../../dataset/SBU/original/lssd102.jpg"


#define MAXLINE 4096

#define DISPLAY_FACTOR 1.0

#define MAXCHANNEL 3

#define MAXF 4
const int u[] = {0, 1, 0, -1, -1, -1, 1, 1};
const int v[] = {-1, 0, 1, 0, 1, 1, -1, -1};


void show(Mat showimg, const char *title = "image", int wait = 0)
{
	Mat img = showimg.clone();
	resize(img, img, Size(0, 0), DISPLAY_FACTOR, DISPLAY_FACTOR, INTER_NEAREST);
	imshow(title, img);
	if (wait) waitKey();
}

void showlp(Mat img, const char *title = "image", int wait = 0)
{
	Mat A = img.clone();
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			auto &pixel = A.at<Vec3f>(i, j);
			for (int c = 0; c < 3; c++) {
				//pixel[c] = pixel[c] < 0 ? 1 + pixel[c] : pixel[c];
				pixel[c] = 0.5 + pixel[c] / 2.0 * 10;
			}
			//pixel[0] = pixel[1] = pixel[2] = pixel[0];
		}
	}
	show(A, title, wait);
}

Mat applymask(Mat img, Mat mask, Scalar colorkey = Scalar(0, 0, 0))
{
	Mat ret(img.size(), CV_8UC3, colorkey);
	img.copyTo(ret, mask);
	return ret;
}

double color_distance(Vec3d a, Vec3d b)
{
	// a dot b = 2 * |a| * |b| * cosab
	double cosab = a.dot(b) / (norm(a) * norm(b));
	return 1 - fabs(cosab);
}

double get_median(vector<double> a)
{
	assert(a.size() > 0);
	int nth = a.size() / 2;
	nth_element(a.begin(), a.begin() + nth, a.end());
	return a[nth];
}

double g_seed_maxtolerance;

Mat get_shadow_seed(Mat img, Point seedpoint)
{
	Mat seedimg;
#define SHADOW_SEED_RESIZEFACTOR 4
#define SHADOW_SEED_RESIZEFACTOR_INV (1.0 / SHADOW_SEED_RESIZEFACTOR)
	Point seedpoint_r = seedpoint * SHADOW_SEED_RESIZEFACTOR_INV;
	seedimg = img.clone();
	resize(seedimg, seedimg, Size(0, 0), SHADOW_SEED_RESIZEFACTOR_INV, SHADOW_SEED_RESIZEFACTOR_INV, INTER_LINEAR);
	Mat mask(seedimg.rows + 2, seedimg.cols + 2, CV_8UC1, Scalar(0));
	Scalar t = Scalar(g_seed_maxtolerance, g_seed_maxtolerance, g_seed_maxtolerance);
	floodFill(seedimg, mask, seedpoint_r, 0, 0, t, t, 4 | (255 << 8) | FLOODFILL_MASK_ONLY | FLOODFILL_FIXED_RANGE);
	mask = Mat(mask, Rect(1, 1, mask.cols - 2, mask.rows - 2));
	resize(mask, mask, Size(0, 0), 1.0 / SHADOW_SEED_RESIZEFACTOR_INV, 1.0 / SHADOW_SEED_RESIZEFACTOR_INV, INTER_LINEAR);
	threshold(mask, mask, 254, 255, THRESH_BINARY);
	mask.at<uchar>(seedpoint) = 255;
	//show(mask, "aaa", 1);
	return mask;
}


void mat2array(vector<double> d[], Mat img, Mat mask)
{
	for (int c = 0; c < MAXCHANNEL; c++) {
		d[c].clear();
	}
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			auto &pixel = img.at<Vec3b>(i, j);
			for (int c = 0; c < MAXCHANNEL; c++) {
				if (mask.at<uchar>(i, j)) {
					d[c].push_back(pixel[c]);
				}
			}
		}
	}
}
double g_surface_threshold;
Mat get_surface(Mat img, Mat shadowseed)
{
	// get median color
	Vec3d median_color;
	vector<double> d[MAXCHANNEL];
	mat2array(d, img, shadowseed);
	for (int c = 0; c < MAXCHANNEL; c++) {
		median_color[c] = get_median(d[c]);
	}

	Mat distimg(img.size(), CV_64FC1, Scalar(0));
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			auto &pixel = img.at<Vec3b>(i, j);
			distimg.at<double>(i, j) = color_distance(pixel, median_color);
		}
	}

	
	Mat ret(img.size(), CV_8UC1);
	for (int i = 0; i < distimg.rows; i++) {
		for (int j = 0; j < distimg.cols; j++) {
			auto &pixel = distimg.at<double>(i, j);
			ret.at<uchar>(i, j) = (pixel >= g_surface_threshold) ? 0 : 255;
		}
	}

	return ret;
}


void grow_region(Mat img, Mat mask, function<int(Vec3b, Vec3b, int)> similar)
{
	int r = img.rows, c = img.cols;
	assert(mask.rows == r && mask.cols == c);

	Mat origmask = mask.clone();

	deque<pair<pair<int, int>, int> > q;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (mask.at<uchar>(i, j)) {
				q.push_back(make_pair(make_pair(i, j), 0));
			}
		}
	}

	while (!q.empty()) {
		auto cur = q.front(); q.pop_front();
		int i = cur.first.first, j = cur.first.second;
		int d = cur.second;
		Vec3b &curp = img.at<Vec3b>(i, j);

		for (int f = 0; f < MAXF; f++) {
			int ni = i + u[f], nj = j + v[f];
			if (ni >= 0 && ni < r && nj >= 0 && nj < c) {
				uchar &flag = mask.at<uchar>(ni, nj);
				if (!flag) {
					Vec3b &nextp = img.at<Vec3b>(ni, nj);
					if (similar(curp, nextp, d + 1)) {
						flag = 255;
						q.push_back(make_pair(make_pair(ni, nj), d + 1));
					}
				}
			}
		}
	}
}

template<typename T>
double calc_sd(Mat inimg, Mat mask)
{
	vector<T> a;
	Mat img = inimg.clone();

	int r = img.rows, c = img.cols;
	assert(mask.rows == r && mask.cols == c);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (mask.at<uchar>(i, j)) {
				a.push_back(img.at<T>(i, j));
			}
		}
	}

	cv::Scalar mean, stddev;
	meanStdDev(a, mean, stddev);

	return stddev[0];
}

void get_Ms_Ml(Mat &Ms, Mat &Ml, Mat img, Mat img_gray, Mat img_Y, Mat img_YCrCb, Mat shadowseed, Mat surfaceimg)
{
	Mat mask_old = shadowseed.clone(), mask_old2 = shadowseed.clone();
	for (int i = 0; i < 255; i++) {
		Mat mask = mask_old.clone();
		Scalar m = mean(img_YCrCb, mask_old);
		grow_region(img_YCrCb, mask, [=](Vec3b a, Vec3b b, int f){
			return (fabs(a[0] - b[0])) < i || fabs(m[0] - b[0]) < i; // FIXME: should also use Cr and Cb
		});

		double a = calc_sd<uchar>(img_Y, mask);
		double b = calc_sd<uchar>(img_Y, surfaceimg - mask);
		cout << "SD: Ms=" << a << ", Ms=" << b << endl;
		if (a < b) {
			mask_old2 = mask_old;
			mask_old = mask;
		} else {
			break;
		}
		//show(applymask(img, mask2), "Ms (tmp)", 1);
	}
	Ms = mask_old;
	Ml = surfaceimg - Ms;
}

void get_dist_matrix(Mat &d, Mat a)
{
	deque<pair<int, int> > q;
	int r = d.rows, c = d.cols;
	assert(a.rows == r && a.cols == c);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			uchar f = a.at<uchar>(i, j);
			d.at<int>(i, j) = f ? 0 : -1;
			if (f) {
				q.push_back(make_pair(i, j));
			}
		}
	}
	while (!q.empty()) {
		int i = q.front().first, j = q.front().second;
		q.pop_front();
		int dist = d.at<int>(i, j);
		for (int f = 0; f < MAXF; f++) {
			int ni = i + u[f], nj = j + v[f];
			if (0 <= ni && ni < r && 0 <= nj && nj < c) {
				int &ndist = d.at<int>(ni, nj);
				if (ndist < 0) {
					ndist = dist + 1;
					q.push_back(make_pair(ni, nj));
				}
			}
		}
	}
}
Mat get_Mshadow(Mat Ms, Mat Ml)
{
	assert(Ms.size() == Ml.size());
	Mat Mshadow(Ms.size(), CV_8UC1, Scalar(0));

	int r = Ms.rows, c = Ms.cols;
	assert(Ml.rows == r && Ml.cols == c);

	Mat A(Ms.size(), CV_8UC1, Scalar(255));
	deque<pair<int, int> > q;
	for (int i = 0; i < r; i++) {
		q.push_back(make_pair(i, c));
		q.push_back(make_pair(i, -1));
	}
	for (int i = 0; i < c; i++) {
		q.push_back(make_pair(c, i));
		q.push_back(make_pair(-1, i));
	}
	while (!q.empty()) {
		int i = q.front().first, j = q.front().second;
		q.pop_front();
			
		for (int f = 0; f < MAXF; f++) {
			int ni = i + u[f], nj = j + v[f];
			if (0 <= ni && ni < r && 0 <= nj && nj < c) {
				if (!Ms.at<uchar>(ni, nj) && A.at<uchar>(ni, nj)) {
					A.at<uchar>(ni, nj) = 0;
					q.push_back(make_pair(ni, nj));
				}
			}
		}
	}
	int K;

	K = 1; erode(A, A, getStructuringElement(MORPH_RECT, Size(2 * K + 1, 2 * K + 1), Point(K, K)));
	A = A & ~Ml;
	//show(A, "A");

	Mat B;
	
	K = 3; dilate(A, B, getStructuringElement(MORPH_RECT, Size(2 * K + 1, 2 * K + 1), Point(K, K)));
	B = ~B;
	//show(B, "B");

	Mat trimap(Ms.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat inmat[] = {A, B};
	int from_to[] = {0, 2, 1, 1};
	mixChannels(inmat, 2, &trimap, 1, from_to, 2);
	show(trimap, "trimap");

	Mat distA(Ms.size(), CV_32SC1, Scalar(0));
	Mat distB(Ms.size(), CV_32SC1, Scalar(0));
	get_dist_matrix(distA, A);
	get_dist_matrix(distB, B);

	Mshadow = Mat(Ms.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			int dA = distA.at<int>(i, j);
			int dB = distB.at<int>(i, j);
			Mshadow.at<uchar>(i, j) = (dA < dB) ? 255 : 0;
		}
	}

	return Mshadow;
}


void get_Ilit(Mat &Ilit, Mat img, Mat img_Y, Mat Ml, Mat Ms, Mat Mshadow)
{
	double stddev_l = calc_sd<float>(img_Y, Ml);
	double stddev_s = calc_sd<float>(img_Y, Ms);
	double gamma = stddev_l / stddev_s;
	cout << "stddev(Ml)=" << stddev_l << "   stddev(Ms)=" << stddev_s << "   =>  gamma=" << gamma << endl;
	
	Scalar mean_l = mean(img, Ml);
	Scalar mean_s = mean(img, Ms);
	cout << "mean(Ml)=" << mean_l << "  mean(Ms)=" << mean_s << endl;
	
	Scalar alpha;
	for (int c = 0; c < MAXCHANNEL; c++) {
		alpha[c] = mean_l[c] - gamma * mean_s[c];
	}

	Ilit = img.clone();
	for (int i = 0; i < Ilit.rows; i++) {
		for (int j = 0; j < Ilit.cols; j++) {
			if (Mshadow.at<uchar>(i, j)) {
				auto &dst = Ilit.at<Vec3f>(i, j);
				auto &src = img.at<Vec3f>(i, j);
				for (int c = 0; c < MAXCHANNEL; c++) {
					dst[c] = alpha[c] + gamma * src[c];
				}
			}
		}
	}
}


void make_pyramid(Mat img, Mat *gp, Mat *lp, int levels)
{
	// generate Gaussian pyramid
	gp[0] = img.clone();
	for (int i = 1; i < levels; i++) {
		pyrDown(gp[i - 1], gp[i]);
		//show(gp[i], "gp", 1);
	}
	// generate Laplacian Pyramid
	lp[levels - 1] = gp[levels - 1];
	for (int i = levels - 2; i >= 0; i--) {
		Mat ge;
		pyrUp(gp[i + 1], ge);
		lp[i] = gp[i] - ge;
	}
}


Point g_seedpoint;

void run(Mat img)
{
	//////////////// Preprocess

	Mat img_YCrCb, img_gray;
	cvtColor(img, img_YCrCb, COLOR_BGR2YCrCb);
	cvtColor(img, img_gray, CV_BGR2GRAY);
	Mat img_Y(img_YCrCb.size(), CV_8UC1, Scalar(0));
	{
		int from_to[] = {0, 0};
		mixChannels(&img_YCrCb, 1, &img_Y, 1, from_to, 1);
	}
	Mat img_B(img.size(), CV_8UC1, Scalar(0));
	Mat img_G(img.size(), CV_8UC1, Scalar(0));
	Mat img_R(img.size(), CV_8UC1, Scalar(0));

	Mat img_BGR_mats[] = {img_B, img_G, img_R};
	{
		int from_to[] = {0, 0, 1, 1, 2, 2};
		mixChannels(&img, 1, img_BGR_mats, 3, from_to, 3);
	}

	/*show(img_B, "B");
	show(img_G, "G");
	show(img_R, "R");
	show(img_Y, "Y");*/






	//////////////// Shadow Detection


	// get shadow seed
	Mat shadowseed = get_shadow_seed(img, g_seedpoint);
	show(applymask(img, shadowseed), "seed");
	
	// get surface
	Mat surfaceimg = get_surface(img, shadowseed);
	show(applymask(img, surfaceimg), "surface");

	// get Ms and Ml
	Mat Ms, Ml;
	get_Ms_Ml(Ms, Ml, img, img_gray, img_Y, img_YCrCb, shadowseed, surfaceimg);
	show(applymask(img, Ms), "Ms (in shadow and on same surface)");
	show(applymask(img, Ml), "Ml (outside shadow)");

	// get Mshadow
	Mat Mshadow = get_Mshadow(Ms, Ml);
	show(applymask(img, Mshadow), "Mshadow (real shadow)");





	///////////////// Illumination Recovery

	Mat f_img, f_img_Y, f_img_YCrCb;
	img.convertTo(f_img, CV_32FC3, 1.0 / 255.0);
	img_Y.convertTo(f_img_Y, CV_32FC3, 1.0 / 255.0);
	img_YCrCb.convertTo(f_img_YCrCb, CV_32FC3, 1.0 / 255.0);


	// direct method

	Mat Ilit;
	get_Ilit(Ilit, f_img, f_img_Y, Ml, Ms, Mshadow);
	show(Ilit, "Ilit");

	

	// pyramid method

	#define PYRAMID_LEVELS 3

	Mat gp[PYRAMID_LEVELS], lp[PYRAMID_LEVELS];
	make_pyramid(f_img, gp, lp, PYRAMID_LEVELS);

	Mat gp_YCrCb[PYRAMID_LEVELS], lp_YCrCb[PYRAMID_LEVELS];
	make_pyramid(f_img_YCrCb, gp_YCrCb, lp_YCrCb, PYRAMID_LEVELS);

	Mat p_Ml[PYRAMID_LEVELS], p_Ms[PYRAMID_LEVELS], p_Mshadow[PYRAMID_LEVELS];
	p_Ml[0] = Ml.clone();
	p_Ms[0] = Ms.clone();
	p_Mshadow[0] = Mshadow.clone();
	for (int i = 1; i < PYRAMID_LEVELS; i++) {
		pyrDown(p_Ml[i - 1], p_Ml[i]);
		pyrDown(p_Ms[i - 1], p_Ms[i]);
		pyrDown(p_Mshadow[i - 1], p_Mshadow[i]);
	}
	for (int i = 1; i < PYRAMID_LEVELS; i++) {
		double t = 128;
		threshold(p_Ml[i], p_Ml[i], t, 255.0, THRESH_BINARY);
		threshold(p_Ms[i], p_Ms[i], t, 255.0, THRESH_BINARY);
		threshold(p_Mshadow[i], p_Mshadow[i], t, 255.0, THRESH_BINARY);
	}



	Mat lp_out[PYRAMID_LEVELS];
	for (int i = 0; i < PYRAMID_LEVELS; i++) {
		Mat lp_Y(lp_YCrCb[i].size(), CV_32FC1, Scalar(0));
		{
			int from_to[] = {0, 0};
			mixChannels(&lp_YCrCb[i], 1, &lp_Y, 1, from_to, 1);
		}

		Mat A = lp[i].clone();
		Mat B;
		cout << "pyramid level " << i << endl;
		get_Ilit(B, A, lp_Y, p_Ml[i], p_Ms[i], p_Mshadow[i]);
		
		auto showfunc = (i < PYRAMID_LEVELS - 1) ? showlp : show;


		char nameA[MAXLINE];
		char nameB[MAXLINE];
		sprintf(nameA, "A%d", i);
		sprintf(nameB, "B%d", i);
		showfunc(A, nameA, 0);
		showfunc(B, nameB, 0);

		lp_out[i] = B;
	}

	// reconstruct from two pyramids
	Mat rc;
	rc = lp_out[PYRAMID_LEVELS - 1];
	for (int i = PYRAMID_LEVELS - 2; i >= 0; i--) {
		pyrUp(rc, rc);
		rc = rc + lp_out[i];
		gp[i].copyTo(rc, 255 - p_Mshadow[i]);
	}
	show(rc, "pyramid");




	
	

	


}


Mat g_img;


static int surface_threshold_int_max = 10000;
static int surface_threshold_int = surface_threshold_int_max * 0.3;

static int seed_maxtolerance_int = 20;
static int seed_maxtolerance_int_max = 255;

void redraw()
{
	g_surface_threshold = (double) surface_threshold_int / surface_threshold_int_max;
	g_surface_threshold *= 0.1;

	g_seed_maxtolerance = seed_maxtolerance_int;

	cout << "   REDRAW: " << g_seedpoint << "  " << g_surface_threshold << "   " << g_seed_maxtolerance << endl;
	run(g_img);
}

void mousehandler(int k, int x, int y, int s, void *p)
{
	if (k && s) {
		cout << "click: (" << x << ", " << y << ")\n";
		g_seedpoint = Point(x, y) / DISPLAY_FACTOR;
		redraw();
	}
}

int main(int argc, char *argv[])
{
	const char *filename;
#ifndef IMAGE_FILENAME
	assert(argc >= 2);
	filename = argv[1];
#else
	filename = IMAGE_FILENAME;
#endif
	const char *title = "original";
	g_img = imread(filename);
	int nr = g_img.rows;
	int nc = g_img.cols;
	assert((1 << PYRAMID_LEVELS) % SHADOW_SEED_RESIZEFACTOR == 0);
	nr = nr / (1 << PYRAMID_LEVELS) * (1 << PYRAMID_LEVELS);
	nc = nc / (1 << PYRAMID_LEVELS) * (1 << PYRAMID_LEVELS);
	g_img = Mat(g_img, Rect(0, 0, nc, nr));
	
	show(g_img, title);
	setMouseCallback(title, mousehandler);
	createTrackbar("surface", title, &surface_threshold_int, surface_threshold_int_max, [](int, void *){ redraw(); });
	createTrackbar("seed", title, &seed_maxtolerance_int, seed_maxtolerance_int_max, [](int, void *){ redraw();	});

	waitKey();

	/*try {
		run();
	} catch (Exception &e) {
		cout << e.what() << endl;
	}*/

	return 0;
}

