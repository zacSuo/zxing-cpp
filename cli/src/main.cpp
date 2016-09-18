// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  Copyright 2010-2011 ZXing authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <fstream>
#include <string>
#include "ImageReaderSource.h"
#include <zxing/common/Counted.h>
#include <zxing/Binarizer.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/Result.h>
#include <zxing/ReaderException.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/common/HybridBinarizer.h>
#include <exception>
#include <zxing/Exception.h>
#include <zxing/common/IllegalArgumentException.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>

#include <zxing/qrcode/QRCodeReader.h>
#include <zxing/multi/qrcode/QRCodeMultiReader.h>
#include <zxing/multi/ByQuadrantReader.h>
#include <zxing/multi/MultipleBarcodeReader.h>
#include <zxing/multi/GenericMultipleBarcodeReader.h>

#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv.hpp>
#include <cv.h>

#include "opencvbitmapsource.h"

using namespace std;
using namespace zxing;
using namespace zxing::multi;
using namespace zxing::qrcode;
using namespace cv;


namespace {

bool more = false;
bool test_mode = false;
bool try_harder = false;
bool search_multi = false;
bool use_hybrid = false;
bool use_global = false;
bool verbose = false;

}

vector<Ref<Result> > decode(Ref<BinaryBitmap> image, DecodeHints hints) {
  Ref<Reader> reader(new MultiFormatReader);
  return vector<Ref<Result> >(1, reader->decode(image, hints));
}

vector<Ref<Result> > decode_multi(Ref<BinaryBitmap> image, DecodeHints hints) {
  MultiFormatReader delegate;
  GenericMultipleBarcodeReader reader(delegate);
  return reader.decodeMultiple(image, hints);
}

int read_image(Ref<LuminanceSource> source, bool hybrid, string expected) {
  vector<Ref<Result> > results;
  string cell_result;
  int res = -1;

  try {
    Ref<Binarizer> binarizer;
    if (hybrid) {
      binarizer = new HybridBinarizer(source);
    } else {
      binarizer = new GlobalHistogramBinarizer(source);
    }
    DecodeHints hints(DecodeHints::DEFAULT_HINT);
    hints.setTryHarder(try_harder);
    Ref<BinaryBitmap> binary(new BinaryBitmap(binarizer));
    if (search_multi) {
      results = decode_multi(binary, hints);
    } else {
      results = decode(binary, hints);
    }
    res = 0;
  } catch (const ReaderException& e) {
    cell_result = "zxing::ReaderException: " + string(e.what());
    res = -2;
  } catch (const zxing::IllegalArgumentException& e) {
    cell_result = "zxing::IllegalArgumentException: " + string(e.what());
    res = -3;
  } catch (const zxing::Exception& e) {
    cell_result = "zxing::Exception: " + string(e.what());
    res = -4;
  } catch (const std::exception& e) {
    cell_result = "std::exception: " + string(e.what());
    res = -5;
  }


  if (test_mode && results.size() == 1) {
    std::string result = results[0]->getText()->getText();
    if (expected.empty()) {
      cout << "  Expected text or binary data for image missing." << endl
           << "  Detected: " << result << endl;
      res = -6;
    } else {
      if (expected.compare(result) != 0) {
        cout << "  Expected: " << expected << endl
             << "  Detected: " << result << endl;
        cell_result = "data did not match";
        res = -6;
      }
    }
  }

  if (res != 0 && (verbose || (use_global ^ use_hybrid))) {
    cout << (hybrid ? "Hybrid" : "Global")
         << " binarizer failed: " << cell_result << endl;
  } else if (!test_mode) {
    if (verbose) {
      cout << (hybrid ? "Hybrid" : "Global")
           << " binarizer succeeded: " << endl;
    }
    for (size_t i = 0; i < results.size(); i++) {
      if (more) {
        cout << "  Format: "
             << BarcodeFormat::barcodeFormatNames[results[i]->getBarcodeFormat()]
             << endl;
        for (int j = 0; j < results[i]->getResultPoints()->size(); j++) {
          cout << "  Point[" << j <<  "]: "
               << results[i]->getResultPoints()[j]->getX() << " "
               << results[i]->getResultPoints()[j]->getY() << endl;
        }
      }
      if (verbose) {
        cout << "    ";
      }
      cout << results[i]->getText()->getText() << endl;
    }
  }

  return res;
}

string read_expected(string imagefilename) {
  string textfilename = imagefilename;
  string::size_type dotpos = textfilename.rfind(".");

  textfilename.replace(dotpos + 1, textfilename.length() - dotpos - 1, "txt");
  ifstream textfile(textfilename.c_str(), ios::binary);
  textfilename.replace(dotpos + 1, textfilename.length() - dotpos - 1, "bin");
  ifstream binfile(textfilename.c_str(), ios::binary);
  ifstream *file = 0;
  if (textfile.is_open()) {
    file = &textfile;
  } else if (binfile.is_open()) {
    file = &binfile;
  } else {
    return std::string();
  }
  file->seekg(0, ios_base::end);
  size_t size = size_t(file->tellg());
  file->seekg(0, ios_base::beg);

  if (size == 0) {
    return std::string();
  }

  char* data = new char[size + 1];
  file->read(data, size);
  data[size] = '\0';
  string expected(data);
  delete[] data;

  return expected;
}


int main(int argc, char** argv) {
    Mat sourceFrame, grayFrame;
    VideoCapture capture(0);
    if (!capture.isOpened()) return 0;
    bool stopFlag(false );
    while (!stopFlag)
    {
        if (!capture.read(sourceFrame))
        {
            capture.open(0);
            cout<<endl<<capture.isOpened()<< "Camera Read Fail;" <<endl;
            if (!capture.isOpened() || !capture.read(sourceFrame)) break;
        }
		//图像灰度处理
		cvtColor(sourceFrame, grayFrame, CV_BGR2GRAY);
		Ref<OpenCVBitmapSource> source(new OpenCVBitmapSource(grayFrame));
		Ref<Binarizer> binarizer(new GlobalHistogramBinarizer(source));
		Ref<BinaryBitmap> bitmap(new BinaryBitmap(binarizer));
		MultiFormatReader reader;
		Ref<Result> result;
		string strResult;
		try
		{
			result = reader.decode(bitmap, DecodeHints(DecodeHints::TRYHARDER_HINT));
			strResult = result->getText()->getText();
			cout<<"===================================" << strResult <<endl;
		}
		catch(const std::exception& e)
		{
			std::cerr<<e.what()<<std::endl;
		}
		//if(strResult.size() != 0){
		//	CvFont font;
		//	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX ,1.1,1.0,0,2);
		//	//strResult = CodeFormatConvert::convertUTF8ToGB2312(strResult.data());
		//	putText(sourceFrame,strResult,Point(20,20),font);
		//}
		
		imshow("Image", sourceFrame);

		if (waitKey(10) == 27)
		{//监听到ESC退出
			stopFlag = true;
		}
    }

  return 0;
}
