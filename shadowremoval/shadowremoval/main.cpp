#include "pch.h"

#define IMAGE_FILENAME "../../data/" "shadow.png"
#define DISPLAY_FACTOR 2.0

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
	int nth = a.size() / 2;
	nth_element(a.begin(), a.begin() + nth, a.end());
	return a[nth];
}

double g_seed_maxtolerance;

Mat get_shadow_seed(Mat img, Point seedpoint)
{
	Mat seedimg;
#define SHADOW_SEED_RESIZEFACTOR 0.5
	seedpoint *= SHADOW_SEED_RESIZEFACTOR;
	seedimg = img.clone();
	resize(seedimg, seedimg, Size(0, 0), SHADOW_SEED_RESIZEFACTOR, SHADOW_SEED_RESIZEFACTOR, INTER_LINEAR);
	Mat mask(seedimg.rows + 2, seedimg.cols + 2, CV_8UC1, Scalar(0));
	Scalar t = Scalar(g_seed_maxtolerance, g_seed_maxtolerance, g_seed_maxtolerance);
	floodFill(seedimg, mask, seedpoint, 0, 0, t, t, 4 | (255 << 8) | FLOODFILL_MASK_ONLY | FLOODFILL_FIXED_RANGE);
	mask = Mat(mask, Rect(1, 1, mask.cols - 2, mask.rows - 2));
	resize(mask, mask, Size(0, 0), 1.0 / SHADOW_SEED_RESIZEFACTOR, 1.0 / SHADOW_SEED_RESIZEFACTOR, INTER_NEAREST);


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

double calc_sd(Mat inimg, Mat mask)
{
	vector<uchar> a;
	Mat img;
	cvtColor(inimg, img, CV_BGR2GRAY);
	//cvtColor(inimg, img, CV_BGR2YCrCb);
	//show(img, "img", 1);

	int r = img.rows, c = img.cols;
	assert(mask.rows == r && mask.cols == c);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (mask.at<uchar>(i, j)) {
				a.push_back(img.at<uchar>(i, j));
				//a.push_back(img.at<Vec3b>(i, j)[0]);
			}
		}
	}

	cv::Scalar mean, stddev;
	meanStdDev(a, mean, stddev);

	return stddev[0];
}

void get_Ms_Ml(Mat &Ms, Mat &Ml, Mat img, Mat img_YCrCb, Mat shadowseed, Mat surfaceimg)
{
	Ms = shadowseed;
	for (int i = 0; i < 255; i++) {
		Mat mask = Ms.clone();
		Scalar mean_color = mean(img_YCrCb, Ms);
		grow_region(img_YCrCb, mask, [=](Vec3b a, Vec3b b, int f){
			return (fabs(a[0] - b[0])) < i /*|| fabs(mean_color[0] - b[0]) < i*/; // FIXME: should also use Cr and Cb
		});
		Mat mask2 = mask.clone();
		/*grow_region(img_YCrCb, mask, [=](Vec3b a, Vec3b b, int f){
			return f < 3 && fabs(a[0] - b[0]) < i; // FIXME: should also use Cr and Cb
		});*/

		double a = calc_sd(img, mask);
		double b = calc_sd(img, surfaceimg - mask);
		cout << "SD: Ms=" << a << ", Ms=" << b << endl;
		if (a < b) {
			Ms = mask2;
		} else {
			break;
		}
		//show(applymask(img, mask2), "Ms (tmp)", 1);
	}
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

Point g_seedpoint;

void run(Mat img)
{
	Mat img_YCrCb;
	cvtColor(img, img_YCrCb, COLOR_BGR2YCrCb);

	// get shadow seed
	Mat shadowseed = get_shadow_seed(img, g_seedpoint);
	show(applymask(img, shadowseed), "shadow seed");
	
	// get surface
	Mat surfaceimg = get_surface(img, shadowseed);
	show(applymask(img, surfaceimg), "surface");

	// get Ms and Ml
	Mat Ms, Ml;
	get_Ms_Ml(Ms, Ml, img, img_YCrCb, shadowseed, surfaceimg);
	show(applymask(img, Ms), "Ms (in shadow and on same surface)");
	show(applymask(img, Ml), "Ml (outside shadow)");

	// get Mshadow
	Mat Mshadow = get_Mshadow(Ms, Ml);
	show(applymask(img, Mshadow), "Mshadow (real shadow)");


	show(applymask(img, 255 - Mshadow), "final");
}


Mat g_img;


static int surface_threshold_int_max = 10000;
static int surface_threshold_int = surface_threshold_int_max * 0.1;

static int seed_maxtolerance_int = 40;
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

int main()
{
	const char *title = "original";
	g_img = imread(IMAGE_FILENAME);
	int nr = g_img.rows;
	int nc = g_img.cols;
	nr = nr / 2 * 2;
	nc = nc / 2 * 2;
	g_img = Mat(g_img, Rect(0, 0, nc, nr));
	
	show(g_img, title);
	setMouseCallback(title, mousehandler);
	createTrackbar("surface", title, &surface_threshold_int, surface_threshold_int_max, [](int, void *){ redraw(); });
	createTrackbar("seed", title, &seed_maxtolerance_int, seed_maxtolerance_int_max, [](int, void *){ redraw();	});

	waitKey();
	/*try {
		run();
	} catch (Exception &e) {
		MessageBoxA(NULL, e.what(), "exception", 0);
	}*/
	return 0;
}

