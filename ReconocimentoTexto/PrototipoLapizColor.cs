using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;

using Emgu.CV.CvEnum;

using Emgu.CV.Util;

using System.Diagnostics;

using System.Runtime.InteropServices;
using System.Windows;




namespace ReconocimentoTexto
{

    public partial class Form1 : Form
    {
        VideoCapture capture;
        private static Mat imagen = new Mat();
        private static Mat imagenOut = new Mat();
        bool Pause = false;
        //Mat picture = new Mat(@"Dr.JekyllandMr.HydeText.jpg"); // Pick some path on your disk!
        Mat picture = new Mat(); // Pick some path on your disk!
        // Determines boundary of brightness while turning grayscale image to binary (black-white) image
        private const int Threshold = 1;

        // Erosion to remove noise (reduce white pixel zones)
        private const int ErodeIterations = 1;

        // Dilation to enhance erosion survivors (enlarge white pixel zones)
        private const int DilateIterations = 7;

        private static MCvScalar drawingColor = new Bgr(Color.Red).MCvScalar;

        public DateTime prevTime;
        public DateTime currentTime;
        public DateTime changeTime;
        //public Frame currentFrame;
        // public Frame prevFrame;
        int xFinal = 0;
        int yFinal = 0;
        public float leapStart;
        public float leapEnd;
        public float appEnd;
        public float appStart;
        public float leapStarty;
        public float leapEndy;
        public float appEndy;
        public float appStarty;
        bool clickDown = true;
        public Form1()
        {
            InitializeComponent();
        }

        private async void playToolStripMenuItem1_Click(object sender, EventArgs e)//Procedimiento que abre la camara
        {
            if (capture==null)
            {
               // int cameraNumber = (int)numericUpDown1.Value;
                capture = new VideoCapture((int)numericUpDown1.Value);
            }
                
            
            //capture.ImageGrabbed += Capture_ImageGrabbed1;
            capture.Start();
            if (capture == null)
            {
                return;
            }

            try
            {
                while (!Pause)
                {
                    Mat m = new Mat();
                    capture.Read(m);

                    if (!m.IsEmpty)
                    {
                        pictureBox1.Image = m.Bitmap;
                        //double fps = capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps);
                        double fps = 15;
                        await Task.Delay(1000 / Convert.ToInt32(fps));

                    }
                    else
                    {
                        break;
                    }



                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);

            }
        }

        private void Capture_ImageGrabbed1(object sender, EventArgs e)
        {
            try
            {
                Mat m = new Mat();
                capture.Retrieve(m);
                pictureBox1.Image = m.ToImage<Bgr, byte>().Bitmap;

            }
            catch (Exception)
            {

                throw;
            }

            

        }

        private async void detectarManoToolStripMenuItem_ClickAsync(object sender, EventArgs e)
        {
            if (capture == null)
            {
                return;
            }

            try
            {
                while (!Pause)
                {
                    Mat m = new Mat();
                    Mat n = new Mat();
                    Mat o = new Mat();
                    Mat binaryDiffFrame = new Mat();
                    Mat denoisedDiffFrame = new Mat();
                    Mat finalFrame = new Mat();
                    Rectangle cropbox = new Rectangle(); 
                    //pictureBox3.DrawToBitmap();
                    capture.Read(m);

                    if (!m.IsEmpty)
                    {


                        Image<Bgr, byte> ret = m.ToImage<Bgr, byte>();
                        Image<Bgr, byte> img = m.ToImage<Bgr, byte>();
                        var image = img.InRange(new Bgr(trackBar1.Value, trackBar2.Value, trackBar3.Value), new Bgr(trackBar6.Value, trackBar5.Value, trackBar4.Value));
                        var mat = img.Mat;//nueva matriz igual a la anterior
                        mat.SetTo(new MCvScalar(0, 0, 255), image);
                        mat.CopyTo(ret);
                        //return ret;
                        Image<Bgr, byte> imgout = ret.CopyBlank();//imagen sin fondo negro
                        imgout._Or(img);
                        //pictureBox1.Image = img.Bitmap;
                        pictureBox7.Image = imgout.Bitmap;

                        CvInvoke.AbsDiff(m, imgout, n);
                        // Apply binary threshold to grayscale image (white pixel will mark difference)
                        CvInvoke.CvtColor(n, o, ColorConversion.Bgr2Gray);
                        CvInvoke.Threshold(o, binaryDiffFrame, 5, 255, ThresholdType.Binary);// 5 Determines boundary of brightness while turning grayscale image to binary (black-white) image

                        // Remove noise with opening operation (erosion followed by dilation)
                        CvInvoke.Erode(binaryDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), ErodeIterations, BorderType.Default, new MCvScalar(1));
                        CvInvoke.Dilate(denoisedDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), DilateIterations, BorderType.Default, new MCvScalar(1));
                        pictureBox4.Image = denoisedDiffFrame.Bitmap;

                        //Image<Bgr, Byte> imgeOrigenal = BackgroundToGreen(m.ToImage<Bgr, Byte>());
 
                        label5.Text = trackBar1.Value.ToString();
                        label7.Text = trackBar2.Value.ToString();
                        label9.Text = trackBar3.Value.ToString();
                        label6.Text = trackBar6.Value.ToString();
                        label8.Text = trackBar5.Value.ToString();
                        label10.Text = trackBar4.Value.ToString();
                        //DetectarTexto(m.ToImage<Bgr, byte>());
                        //double fps = capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps);
                        m.CopyTo(finalFrame);
                        
                        DetectObject(denoisedDiffFrame, finalFrame, cropbox);


                        pictureBox5.Image = finalFrame.Bitmap;
                        
                        double fps = 15;
                        await Task.Delay(1000 / Convert.ToInt32(fps));

                    }
                    else
                    {
                        break;
                    }
                    

                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);

            }

        }



        private  void DetectObject(Mat detectionFrame, Mat displayFrame,Rectangle box)
        {
            Image<Bgr, Byte> buffer_im = displayFrame.ToImage<Bgr, Byte>();
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                VectorOfPoint biggestContour = null;
                IOutputArray hirarchy = null;
                // Build list of contours
                CvInvoke.FindContours(detectionFrame, contours, hirarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

                // Selecting largest contour
                if (contours.Size > 0)
                {
                    double maxArea = 0;
                    int chosen = 0;
                    VectorOfPoint contour = null;
                    for (int i = 0; i < contours.Size; i++)
                    {
                        contour = contours[i];

                        double area = CvInvoke.ContourArea(contour);
                        if (area > maxArea)
                        {
                            maxArea = area;
                            chosen = i;
                        }
                    }

                    // Draw on a frame
                    //MarkDetectedObject(displayFrame, contours[chosen], maxArea);//dibuja una envoltura roja

                    VectorOfPoint hullPoints = new VectorOfPoint();
                    VectorOfInt hullInt = new VectorOfInt();

                    CvInvoke.ConvexHull(contours[chosen], hullPoints, true);
                    CvInvoke.ConvexHull(contours[chosen], hullInt, false);

                    Mat defects = new Mat();
                    

                    if (hullInt.Size > 3)
                        CvInvoke.ConvexityDefects(contours[chosen], hullInt, defects);

                    box = CvInvoke.BoundingRectangle(hullPoints);
                    CvInvoke.Rectangle(displayFrame, box, drawingColor);//Box rectangulo que encierra el area mas grande
                                                                        // cropbox = crop_color_frame(displayFrame, box);
                    

                    Point center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);//centro  rectangulo MOUSE

                    VectorOfPoint start_points = new VectorOfPoint();
                    VectorOfPoint far_points = new VectorOfPoint();

                    if (!defects.IsEmpty)
                    {
                        //Data from Mat are not directly readable so we convert it to Matrix<>
                        Matrix<int> m = new Matrix<int>(defects.Rows, defects.Cols,
                           defects.NumberOfChannels);
                        defects.CopyTo(m);
                        int xe = 2000, ye = 2000;
                        int xs = 2000, ys = 2000;
                        int xer = 2000, yer = 2000;
                        int xsr = 2000, ysr = 2000;
                        int xem = 0, yem = 0;
                        int xsm = 0, ysm = 0;
                        int xez = 0, yez = 0;
                        int xsz = 0, ysz = 0;
                        int y = 0, x = 0;
                        int ym = 0, xm = 0;
                        int yr = 0, xr = 0;
                        int yz = 0, xz = 0;
                        for (int i = 0; i < m.Rows; i++)
                        {
                            int startIdx = m.Data[i, 0];
                            int endIdx = m.Data[i, 1];
                            int farIdx = m.Data[i, 2];
                            Point startPoint = contours[chosen][startIdx];
                            Point endPoint = contours[chosen][endIdx];
                            Point farPoint = contours[chosen][farIdx];
                            CvInvoke.Circle(displayFrame, endPoint, 3, new MCvScalar(0, 255, 255));
                            CvInvoke.Circle(displayFrame, startPoint, 3, new MCvScalar(255, 255, 0));

                            if (true)
                            {
                                if (endPoint.Y < ye)
                                {
                                    xe = endPoint.X;

                                    ye = endPoint.Y;

                                }

                                if (startPoint.Y < ys)
                                {
                                    xs = startPoint.X;

                                    ys = startPoint.Y;


                                }

                                if (ye < ys)
                                {
                                    y = ye;
                                    x = xe;



                                }
                                else
                                {
                                    y = ys;
                                    x = xs;
                                }


                                if (endPoint.Y > yem)
                                {
                                    xem = endPoint.X;

                                    yem = endPoint.Y;

                                }

                                if (startPoint.Y > ysm)
                                {
                                    xsm = startPoint.X;

                                    ysm = startPoint.Y;


                                }

                                if (yem > ysm)
                                {
                                    ym = yem;
                                    xm = xem;



                                }
                                else
                                {
                                    y = ys;
                                    x = xs;
                                }

                                if (endPoint.X < xer)
                                {
                                    xer = endPoint.X;

                                    yer = endPoint.Y;

                                }

                                if (startPoint.X < xsr)
                                {
                                    xsr = startPoint.X;

                                    ysr = startPoint.Y;


                                }

                                if (xer < xsr)
                                {
                                    yr = yer;
                                    xr = xer;



                                }
                                else
                                {
                                    yr = ysr;
                                    xr = xsr;
                                }


                                if (endPoint.X > xez)
                                {
                                    xez = endPoint.X;

                                    yez = endPoint.Y;

                                }

                                if (startPoint.X > xsz)
                                {
                                    xsz = startPoint.X;

                                    ysz = startPoint.Y;


                                }

                                if (xez > xsz)
                                {
                                    yz = yez;
                                    xz = xez;



                                }

                                else
                                {
                                    yz = ysz;
                                    xz = xsz;
                                
                            

                            }


                        }
                            /*var info = new string[] {
                
                $"Posicion: {endPoint.X}, {endPoint.Y}"
            };

                            WriteMultilineText(displayFrame, info, new Point(endPoint.X + 5, endPoint.Y));*/

                            double distance = Math.Round(Math.Sqrt(Math.Pow((center.X - farPoint.X), 2) + Math.Pow((center.Y - farPoint.Y), 2)), 1);
                            if (distance < box.Height * 0.3)
                            {
                                CvInvoke.Circle(displayFrame, farPoint, 3, new MCvScalar(255, 0, 0));
                            }
                            //draw  a line connecting the convexity defect start point and end point in thin red line
                            CvInvoke.Line(displayFrame, startPoint, endPoint, new MCvScalar(0, 255, 0));
                            // CvInvoke.Line(displayFrame, startPoint, farPoint, new MCvScalar(0, 255, 255));
                        }
                        var infoe = new string[] {$"Punto", $"Posicion: {x}, {y}"};
                        var infos = new string[] { $"Punto", $"Posicion: {xm}, {ym}" };
                        var infor = new string[] { $"Punto", $"Posicion: {x}, {y}" };
                        var infoz = new string[] { $"Punto", $"Posicion: {xm}, {ym}" };
                        var infoCentro = new string[] { $"Centro", $"Posicion: {xm}, {ym}" };

                        var xCentro = (x + xm + xr + xz) / 4;
                        var yCentro = (y + ym + yr + yz) / 4;

                        WriteMultilineText(displayFrame, infoe, new Point(x + 30, y));
                        CvInvoke.Circle(displayFrame, new Point(x, y), 5, new MCvScalar(255, 0, 255), 2);
                        Image<Bgr, byte> temp = detectionFrame.ToImage<Bgr,byte>();
                        var temp2 = temp.SmoothGaussian(5).Convert<Gray, byte>().ThresholdBinary(new Gray(230), new Gray(255));
                        VectorOfVectorOfPoint contorno = new VectorOfVectorOfPoint();
                        Mat mat = new Mat();
                        CvInvoke.FindContours(temp2,contorno, mat,Emgu.CV.CvEnum.RetrType.External,Emgu.CV.CvEnum.ChainApproxMethod.LinkRuns);
                        for (int i = 0; i < contorno.Size; i++)
                        {
                            double perimetro = CvInvoke.ArcLength(contorno[i], true);
                            VectorOfPoint approx = new VectorOfPoint();
                            CvInvoke.ApproxPolyDP(contorno[i],approx,0.04 * perimetro,true);
                            CvInvoke.DrawContours(displayFrame, contorno, i ,new MCvScalar(0,255,255),2);

                        }
                        
                        WriteMultilineText(displayFrame, infos, new Point(xm + 30, ym));
                        CvInvoke.Circle(displayFrame, new Point(xm, ym), 5, new MCvScalar(255, 0, 255), 2);
                        WriteMultilineText(displayFrame, infor, new Point(xr + 30, yr));
                        CvInvoke.Circle(displayFrame, new Point(xr, yr), 5, new MCvScalar(255, 0, 255), 2);
                        WriteMultilineText(displayFrame, infoz, new Point(xz + 30, yz));
                        CvInvoke.Circle(displayFrame, new Point(xz, yz), 5, new MCvScalar(255, 0, 255), 2);

                        WriteMultilineText(displayFrame, infoz, new Point(xCentro + 30, yCentro));
                        CvInvoke.Circle(displayFrame, new Point(xCentro, yCentro), 2, new MCvScalar(0, 100, 0), 4);
                        //CvInvoke.Circle(picture, new Point(x * 2, y * 4), 20, new MCvScalar(255, 0, 255), 2);*/
                        leapStart = (float)numericUpDown9.Value;
                        leapEnd = (float)numericUpDown10.Value;
                        appStart = (float)numericUpDown11.Value;
                        appEnd = (float)numericUpDown12.Value;

                        

                        leapStarty = (float)numericUpDown17.Value;
                        leapEndy = (float)numericUpDown18.Value;
                        appStarty = (float)numericUpDown15.Value;
                        appEndy = (float)numericUpDown16.Value;
                        if (checkBox4.Checked)
                        {
                            xFinal = (int)Math.Abs((xCentro - leapStart) * ((appEnd - appStart) / (leapEnd - leapStart)) + appStart);
                        }
                        else
                        {
                            xFinal = xCentro;
                            //x = (int)(xScreenIntersect * (float)numericUpDown5.Value);
                            //textBox1.AppendText("Cambio de posicion de x1 " + (int)appStart + (float)numericUpDown11.Value);// valor de appstart y numericupdown11
                            label15.Text = xCentro.ToString();
                            

                        }

                        if (checkBox5.Checked)
                        {
                            yFinal = (int)Math.Abs((yCentro - leapStarty) * ((appEndy - appStarty) / (leapEndy - leapStarty)) + appStarty);
                        }
                        else
                        {
                            //y = (int)((float)numericUpDown6.Value - (yScreenIntersect * (float)numericUpDown6.Value) + (float)numericUpDown14.Value);
                            //y = (int)(yScreenIntersect * (float)numericUpDown6.Value + (float)numericUpDown14.Value);
                            yFinal = 500;
                            label16.Text = yCentro.ToString();

                        }

                        if (checkBox5.Checked&& checkBox4.Checked)
                        {
                            MouseCursor.MoveCursor(xFinal, yFinal);
                        }

                       
                        
                    }
                    


                }
                

                

            }
            

        }

        private static void MarkDetectedObject(Mat frame, VectorOfPoint contour, double area)
        {
            // Getting minimal rectangle which contains the contour
            Rectangle box = CvInvoke.BoundingRectangle(contour);
            
            // Drawing contour and box around it
            CvInvoke.Polylines(frame, contour, true, drawingColor, 1,LineType.FourConnected);
            CvInvoke.Rectangle(frame, box, drawingColor);

            // Write information next to marked object
            Point center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);

            var info = new string[] {
                $"Area: {area}",
                $"Position: {center.X}, {center.Y}"
            };

            WriteMultilineText(frame, info, new Point(box.Right + 5, center.Y));
        }

        private static void WriteMultilineText(Mat frame, string[] lines, Point origin)
        {
            for (int i = 0; i < lines.Length; i++)
            {
                int y = i * 10 + origin.Y; // Moving down on each line
                CvInvoke.PutText(frame, lines[i], new Point(origin.X, y), FontFace.HersheyPlain, 0.8, drawingColor);
            }
        }

        public static Image<Bgr, byte> BackgroundToGreen(Image<Bgr, byte> rgbimage)
        {
            for (int i = 0; i < rgbimage.ManagedArray.GetLength(0); i++)
            {
                for (int j = 0; j < rgbimage.ManagedArray.GetLength(1); j++)
                {
                    Bgr currentColor = rgbimage[i, j];

                    if (/*currentColor.Blue >= minB && currentColor.Blue <= maxB &&*/ currentColor.Green >= 255 && 0 <= currentColor.Green /*&& currentColor.Red >= minR && currentColor.Red <= maxR*/)
                    {
                        rgbimage[i, j] = new Bgr(255, 255, 255);
                    }
                }
            }
            return rgbimage;
            /*
            Image<Bgr, byte> ret = rgbimage;
            var image = rgbimage.InRange(new Bgr(190, 190, 190), new Bgr(255, 255, 255));
            var mat = rgbimage.Mat;
            mat.SetTo(new MCvScalar(200, 237, 204), image);
            mat.CopyTo(ret);
            return ret;*/
        }



        class MouseCursor
        {
            [DllImport("user32.dll")]

            private static extern bool SetCursorPos(int x, int y);

            public static void MoveCursor(int x, int y)
            {

                SetCursorPos(x, y);
            }

        }
        class Clicking
        {
            [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
            public static extern void mouse_event(int dwFlags, int dx, int dy, int cButtons, int dwExtraInfo);
            //[DllImport("user32.dll")]
            //static extern void Mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
            //private const UInt32 MOUSEEVENTF_LEFTDOWN = 0x0002;
            //private const UInt32 MOUSEEVENTF_LEFTUP = 0x0004;
            /*private static extern void Mouse_event(
                   UInt32 dwFlags, // motion and click options
                   UInt32 dx, // horizontal position or change
                   UInt32 dy, // vertical position or change
                   UInt32 dwData, // wheel movement
                   IntPtr dwExtraInfo // application-defined information
            );*/

            // public static void SendClick(Point location)
            public static void SendClick(int x, int y)
            {
                // Cursor.Position = location;
                //Mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, new System.IntPtr());
                //Mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, new System.IntPtr());
                mouse_event(0x0002, 0, x, y, 0);
            }
            public static void SendUpClick(int x, int y)
            {
                // Cursor.Position = location;
                //Mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, new System.IntPtr());
                //Mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, new System.IntPtr());
                mouse_event(0x0004, 0, x, y, 0);
            }


        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox1.Checked)
            {
                trackBar1.Value = 0;
                trackBar2.Value = 0;
                trackBar3.Value = 120;
                trackBar6.Value = 125;
                trackBar5.Value = 255;
                trackBar4.Value = 255;

            }
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox2.Checked)
            {
                trackBar1.Value =0;
                trackBar2.Value = 0;
                trackBar3.Value = 230;
                trackBar6.Value = 255;
                trackBar5.Value = 255;
                trackBar4.Value = 255;

            }
        }






    }
}
