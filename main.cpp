#include <stdio.h>
#include "tiffio.h"
#include <memory>

#pragma warning(disable : 4996)
//#define TILED

int main() {
	TIFF* tif = TIFFOpen("pumpkinhill.tif", "r");
#ifdef TIF2
	TIFF* tif2 = TIFFOpen("pumpCopy.tif", "w");
#endif
	uint32 imageWidth, imageHeight;
	uint32 tileWidth, tileLength, tileoffsets, tilebytecounts;
	uint32 sampleperpixel = 4;
	uint32 x, y;

	int dircount = 0;
	do {
		dircount++;
	} while (TIFFReadDirectory(tif));
	printf("%d directories\n", dircount);
	//exit(1);

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imageWidth);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imageHeight);

#ifdef TIF@
	TIFFSetField(tif2, TIFFTAG_IMAGEWIDTH, &imageWidth);
	TIFFSetField(tif2, TIFFTAG_IMAGELENGTH, &imageHeight);
	TIFFSetField(tif2, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel);   // set number of channels per pixel
	TIFFSetField(tif2, TIFFTAG_BITSPERSAMPLE, 8);    // set the size of the channels
	TIFFSetField(tif2, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
	TIFFSetField(tif2, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tif2, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
#endif

#ifdef TILED
	tdata_t buf;
	TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
	TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileLength);
	//TIFFGetField(tif, TIFFTAG_TILELENGTH, &sampleperpixel);
	TIFFGetField(tif, TIFFTAG_TILEOFFSETS, &tileoffsets);
	TIFFGetField(tif, TIFFTAG_TILEBYTECOUNTS, &tilebytecounts);

	TIFFSetField(tif2, TIFFTAG_TILEWIDTH, &tileWidth);
	TIFFSetField(tif2, TIFFTAG_TILELENGTH, &tileLength);
	TIFFSetField(tif2, TIFFTAG_TILEOFFSETS, &tileoffsets);
	TIFFSetField(tif2, TIFFTAG_TILEBYTECOUNTS, &tilebytecounts);

	buf = _TIFFmalloc(TIFFTileSize(tif));
	for (y = 0; y < imageHeight; y += tileLength)
		for (x = 0; x < imageWidth; x += tileWidth) {
			TIFFReadTile(tif, buf, x, y, 0, 0);
			//TIFFWriteTile(tif2, buf, x, y, 0, 0);
		}
#elif TIF2

	char* image = new char[imageWidth * imageHeight * sampleperpixel];

	tsize_t linebytes = sampleperpixel * imageWidth;     // length in memory of one row of pixel in the image.

	unsigned char* buf = NULL;        // buffer used to store the row of pixel information for writing to file
	//    Allocating memory to store the pixels of current row
	if (TIFFScanlineSize(tif2) < linebytes)
		buf = (unsigned char*)_TIFFmalloc(linebytes);
	else
		buf = (unsigned char*)_TIFFmalloc(TIFFScanlineSize(tif2));

	// We set the strip size of the file to be size of one row of pixels
	TIFFSetField(tif2, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif2, imageWidth * sampleperpixel));

	//Now writing image to the file one strip at a time
	for (uint32 row = 0; row < imageHeight; row++)
	{
		memcpy(buf, &image[(imageHeight - row - 1) * linebytes], linebytes);    // check the index here, and figure out why not using h*linebytes
		if (TIFFWriteScanline(tif2, buf, row, 0) < 0)
			break;
	}
	_TIFFfree(buf);
	TIFFClose(tif2);
#endif


	unsigned char header[54];
	memset(header, 0, 54);
	header[0] = 'B';
	header[1] = 'M';
	int temp;
	temp = imageHeight * imageWidth * 4 + 54;
	memcpy(&header[2], &temp, 4);

	temp = 54;
	memcpy(&header[10], &temp, 4);

	temp = 40;
	memcpy(&header[14], &temp, 4);

	temp = imageHeight;
	memcpy(&header[18], &temp, 4);
	temp = imageWidth;
	memcpy(&header[22], &temp, 4);
	short temps = 1;
	memcpy(&header[26], &temps, 2);
	temps = 4 * 8;
	memcpy(&header[28], &temps, 2);

	temp = 100;
	memcpy(&header[38], &temp, 4);
	memcpy(&header[42], &temp, 4);

	FILE* file = fopen("pumpCopy.bmp", "wb");

	fwrite(header, 1, 54, file);

	uint32 * raster = (uint32*)_TIFFmalloc(imageHeight * imageWidth * sizeof(uint32));
	if (raster != NULL) {
		if (TIFFReadRGBAImage(tif, imageWidth, imageHeight, raster, 0)) {
			printf("HERE\n");
			fwrite(raster, 1, imageHeight * imageWidth * sizeof(uint32), file);
		}
		_TIFFfree(raster);
	}

	fclose(file);
	TIFFClose(tif);
}