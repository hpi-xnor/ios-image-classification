//
//  ViewController.m
//  WhatIsThis
//
//  Created by Haoxiang Li on 1/23/16.
//  Copyright © 2016 Haoxiang Li. All rights reserved.
//

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <AVFoundation/AVCaptureDevice.h> // For access to the camera
#import <AVFoundation/AVCaptureInput.h> // For adding a data input to the camera
#import <AVFoundation/AVCaptureSession.h>

#include <queue>

//NSLock *predictionRunningLock = [NSLock new] ;

dispatch_semaphore_t predictionRunningSemaphore = dispatch_semaphore_create(1);

static void * ExposureTargetBiasContext = &ExposureTargetBiasContext;

@interface ViewController () <UIImagePickerControllerDelegate, UINavigationControllerDelegate>

@property (nonatomic, retain) UIActivityIndicatorView *indicatorView;

@end

@implementation ViewController


- (NSString *)classifyNumber:(UIImage *)image {
    const int numForRendering = kDefaultWidth*kDefaultHeight*(kDefaultChannels+1);
    const int numForComputing = kDefaultWidth*kDefaultHeight*kDefaultChannels;
    
    image = [self rotate: [self squareImage:image] by:90];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    uint8_t imageData[numForRendering];
    CGContextRef contextRef = CGBitmapContextCreate(imageData,
                                                    kDefaultWidth,
                                                    kDefaultHeight,
                                                    8,
                                                    kDefaultWidth*(kDefaultChannels+1),
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, kDefaultWidth, kDefaultHeight), image.CGImage);
//    
//    CGImageRef imgRef = CGBitmapContextCreateImage(contextRef);
//    UIImage* img = [UIImage imageWithCGImage:imgRef];
//    CGImageRelease(imgRef);
    
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    //< Subtract the mean and copy to the input buffer
    std::vector<float> input_buffer(numForComputing);
    float *p_input_buffer[3] = {
        input_buffer.data(),
        input_buffer.data() + kDefaultWidth*kDefaultHeight,
        input_buffer.data() + kDefaultWidth*kDefaultHeight*2};
    const float p_mean[3] = {123.68, 116.779, 103.939};
    for (int i = 0, map_idx = 0, glb_idx = 0; i < kDefaultHeight; i++) {
        for (int j = 0; j < kDefaultWidth; j++) {
            p_input_buffer[0][map_idx] = imageData[glb_idx++] - p_mean[0];
            p_input_buffer[1][map_idx] = imageData[glb_idx++] - p_mean[1];
            p_input_buffer[2][map_idx] = imageData[glb_idx++] - p_mean[2];
            glb_idx++;
            map_idx++;
        }
    }
    
    mx_uint *shape = nil;
    mx_uint shape_len = 0;
    MXPredSetInput(predictor, "data", input_buffer.data(), numForComputing);
    
    NSDate *methodStart = [NSDate date];
    MXPredForward(predictor);
    NSDate *methodFinish = [NSDate date];
    
    MXPredGetOutputShape(predictor, 0, &shape, &shape_len);
    mx_uint tt_size = 1;
    for (mx_uint i = 0; i < shape_len; i++) {
        tt_size *= shape[i];
    }
    std::vector<float> outputs(tt_size);
    MXPredGetOutput(predictor, 0, outputs.data(), tt_size);
    
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    NSString *timeOutput = [NSString stringWithFormat:@"forward pass took %f", executionTime];
    NSLog(@"%@", timeOutput);
    
    dispatch_async(dispatch_get_main_queue(), ^(){
    });
    
    size_t max_idx = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    
    NSString *classification = [[model_synset objectAtIndex:max_idx] componentsJoinedByString:@" "];
    
    std::priority_queue<std::pair<double, int>> q;
    int k = 5; // number of indices we need
    for (int i = 0; i < outputs.size(); ++i) {
        if(q.size()<k)
            q.push(std::pair<double, int>(outputs[i], i));
        else if(q.top().first<outputs[i]){
            q.pop();
            q.push(std::pair<double, int>(outputs[i], i));
        }
    }
    
    NSMutableArray* stringArray = [NSMutableArray new];
    for (int i = 0; i < 5; ++i) {
        [stringArray addObject: [NSString stringWithFormat:@"%f %@", q.top().first, [[model_synset objectAtIndex:q.top().second] componentsJoinedByString:@" "]]];
        q.pop();
    }
    
    dispatch_async(dispatch_get_main_queue(), ^(){
        self.labelTime.text = timeOutput;
        [self.imageViewCrop setImage: image];
        int index = 0;
        for (id obj in top5Labels) {
            ((UILabel*)obj).text = [stringArray objectAtIndex: index];
            index++;
        }
    });
    
    return classification;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    self.indicatorView = [UIActivityIndicatorView new];
    [self.indicatorView setActivityIndicatorViewStyle:UIActivityIndicatorViewStyleGray];
    
    self.labelTime.textAlignment = NSTextAlignmentRight;
    
    top5Labels = [NSArray arrayWithObjects:self.labelTop1, self.labelTop2, self.labelTop3, self.labelTop4, self.labelTop5, nil];
    
    if (!predictor) {
        NSString *jsonPath = [[NSBundle mainBundle] pathForResource:@"binarized_32_resnet-18-binary-symbol.json" ofType:nil];
        NSString *paramsPath = [[NSBundle mainBundle] pathForResource:@"binarized_32_resnet-18-binary-0001.params" ofType:nil];
        NSString *synsetPath = [[NSBundle mainBundle] pathForResource:@"synset.txt" ofType:nil];
        model_symbol = [[NSString alloc] initWithData:[[NSFileManager defaultManager] contentsAtPath:jsonPath] encoding:NSUTF8StringEncoding];
        model_params = [[NSFileManager defaultManager] contentsAtPath:paramsPath];
        
        NSString *input_name = @"data";
        const char *input_keys[1];
        input_keys[0] = [input_name UTF8String];
        const mx_uint input_shape_indptr[] = {0, 4};
        const mx_uint input_shape_data[] = {1, kDefaultChannels, kDefaultWidth, kDefaultHeight};
        MXPredCreate([model_symbol UTF8String], [model_params bytes], (int)[model_params length], 1, 0, 1,
                     input_keys, input_shape_indptr, input_shape_data, &predictor);
        
        model_synset = [NSMutableArray new];
        NSString* synsetText = [NSString stringWithContentsOfFile:synsetPath
                                                         encoding:NSUTF8StringEncoding error:nil];
        NSArray* lines = [synsetText componentsSeparatedByCharactersInSet:
                          [NSCharacterSet newlineCharacterSet]];
        for (NSString *l in lines) {
            NSArray *parts = [l componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            if ([parts count] > 1) {
                [model_synset addObject:[parts subarrayWithRange:NSMakeRange(1, [parts count]-1)]];
            }
        }
        
    }
}

- (UIImage *)cropRect:(UIImage *) image {
    CGRect rect = CGRectMake((image.size.width - image.size.height) / 2,0, image.size.width + (image.size.width - image.size.height) / 2, image.size.height);

    UIGraphicsBeginImageContext(rect.size);
    CGContextRef context = UIGraphicsGetCurrentContext();

    // translated rectangle for drawing sub image
    CGRect drawRect = CGRectMake(-rect.origin.x, -rect.origin.y, image.size.height, image.size.height);

    // clip to the bounds of the image context
    // not strictly necessary as it will get clipped anyway?
    CGContextClipToRect(context, CGRectMake(0, 0, rect.size.height, rect.size.height));

    // draw image
    [image drawInRect:drawRect];

    // grab image
    UIImage* subImage = UIGraphicsGetImageFromCurrentImageContext();

    UIGraphicsEndImageContext();
    
    return subImage;
}

-(UIImage *)squareImage:(UIImage *)image
{
    if (image.size.width<=image.size.height) {
        return nil;
    }
    
    UIGraphicsBeginImageContext(CGSizeMake(image.size.height, image.size.height));
    [image drawInRect:CGRectMake(0, 0, image.size.width, image.size.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

- (UIImage *)rotate:(UIImage *) image by: (CGFloat)degrees {
    CGFloat radians = degrees * (M_PI / 180.0);;
    
    UIView *rotatedViewBox = [[UIView alloc] initWithFrame:CGRectMake(0,0, image.size.width, image.size.height)];
    CGAffineTransform t = CGAffineTransformMakeRotation(radians);
    rotatedViewBox.transform = t;
    CGSize rotatedSize = rotatedViewBox.frame.size;
    
    UIGraphicsBeginImageContextWithOptions(rotatedSize, NO, [[UIScreen mainScreen] scale]);
    CGContextRef bitmap = UIGraphicsGetCurrentContext();
    
    CGContextTranslateCTM(bitmap, rotatedSize.width / 2, rotatedSize.height / 2);
    
    CGContextRotateCTM(bitmap, radians);
    
    CGContextScaleCTM(bitmap, 1.0, -1.0);
    CGContextDrawImage(bitmap, CGRectMake(-image.size.width / 2, -image.size.height / 2 , image.size.width, image.size.height), image.CGImage );
    
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return newImage;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
    NSLog(@"Received Memory Warning!");
}

- (IBAction)selectPhotoButtonTapped:(id)sender {
    UIImagePickerController *imagePicker = [UIImagePickerController new];
    imagePicker.allowsEditing = NO;
    imagePicker.sourceType =  UIImagePickerControllerSourceTypePhotoLibrary;
    imagePicker.delegate = self;
    [self presentViewController:imagePicker animated:YES completion:nil];
}

- (IBAction)capturePhotoButtonTapped:(id)sender {
    UIImagePickerController *imagePicker = [UIImagePickerController new];
    imagePicker.allowsEditing = NO;
    imagePicker.sourceType =  UIImagePickerControllerSourceTypeCamera;
    imagePicker.delegate = self;
    [self presentViewController:imagePicker animated:YES completion:nil];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info {
    UIImage *chosenImage = info[UIImagePickerControllerOriginalImage];
    self.imageViewPhoto.image = chosenImage;
    [picker dismissViewControllerAnimated:YES completion:^(void){
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(){
            [self prepareAndClassify:chosenImage];
        });
    }];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    [picker dismissViewControllerAnimated:YES completion:nil];
}

- (IBAction)startDetectionButtonTapped:(id)sender {
    [self.detectionButton setTitle:@"Stop Detection" forState:UIControlStateNormal];
    [self.detectionButton addTarget:self
                             action:@selector(stopDetectionButtonTapped:)
                   forControlEvents:UIControlEventTouchUpInside];
    
    if (!videoDevice) {
        videoDevice = [self selectCameraAt:AVCaptureDevicePositionBack];
    }
    
    if (!captureSession) {
        captureSession = [self createCaptureSessionFor:videoDevice];
    }
 
    [captureSession startRunning];
}

- (IBAction)stopDetectionButtonTapped:(id)sender {
    [self.detectionButton setTitle:@"Start Detection" forState:UIControlStateNormal];
    [self.detectionButton addTarget:self
                             action:@selector(startDetectionButtonTapped:)
                   forControlEvents:UIControlEventTouchUpInside];
    
    [captureSession stopRunning];
}

- (AVCaptureSession *)createCaptureSessionFor:(AVCaptureDevice *)device
{
    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPresetHigh;
    
    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input) {
        // Handle the error appropriately.
        NSLog(@"no input.....");
    }
    [session addInput:input];
    
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    [session addOutput:output];
    output.videoSettings = @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };
    
    dispatch_queue_t queue = dispatch_queue_create("MyQueue", NULL);
    
    [output setSampleBufferDelegate:self queue:queue];
    
    AVCaptureVideoPreviewLayer *previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:session];
    previewLayer.frame = self.imageViewPhoto.bounds; // Assume you want the preview layer to fill the view.
    [self.imageViewPhoto.layer addSublayer:previewLayer];
    
    return session;
}

- (AVCaptureDevice *)selectCameraAt:(AVCaptureDevicePosition)chosenPosition {
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in devices) {
        if ([device position] == chosenPosition) {
            return device;
        }
    }
    return nil;
}

- (void) prepareAndClassify:(UIImage *) image
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(){
        // classify pic
        NSString *classification = [self classifyNumber:image];
        
        dispatch_semaphore_signal(predictionRunningSemaphore);
    });
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    
    if (dispatch_semaphore_wait(predictionRunningSemaphore, DISPATCH_TIME_NOW) != 0) {
        return;
    }
    
    CGImageRef cgImage = [self imageFromSampleBuffer:sampleBuffer];
    
    UIImage *image = [UIImage imageWithCGImage: cgImage];
    
    [self prepareAndClassify: image];
    
    CGImageRelease(cgImage);
}

- (CGImageRef) imageFromSampleBuffer:(CMSampleBufferRef) sampleBuffer
{
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer,0);
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    CGContextRef newContext = CGBitmapContextCreate(baseAddress, width, height, 8, bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGImageRef newImage = CGBitmapContextCreateImage(newContext);
    CGContextRelease(newContext);
    
    CGColorSpaceRelease(colorSpace);
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    return newImage;
}


@end
