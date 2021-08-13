using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Keras;
using Tensorflow;
using Numpy;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using PandasNet;
using CsvHelper;
using System.IO;
using System.Globalization;
using OxyPlot.WindowsForms;
using OxyPlot;

namespace ANN_Forecast1
{

    class Program
    {
        static void Main(string[] args)
        {
            string df_path = "D:/Documentos/Nuevas descargas/Datasets Solares/muestra_udea(scaled_comas).csv";
            var x_train_list = new List<float>();
            var y_train_list = new List<float>();
            var x_test_list = new List<float>();
            var y_test_list = new List<float>();
            int len = 0;
            var records = new List<Data>();
            using (var reader = new StreamReader(df_path))
            using (CsvReader csv = new CsvReader(reader, CultureInfo.InvariantCulture)) 
            {
                csv.Read();
                csv.ReadHeader();
                while (csv.Read())
                {

                    var record = new Data
                    {
                        POT_filter = csv.GetField<float>("POT_filter"),
                        IRR_filter = csv.GetField<float>("IRR_filter")

                    };
                    records.Add(record);
                    len = len+1;

                }
            }

            int size = (int)(len * 0.8);
            int size_cont = 0;
            foreach (Data i in records) 
            {
                if (size_cont < size)
                {
                    x_train_list.Add(i.IRR_filter);
                    y_train_list.Add(i.POT_filter);
                }
                else
                {
                    x_test_list.Add(i.IRR_filter);
                    y_test_list.Add(i.POT_filter);
                }
                size_cont = size_cont+1;

                
            }
                       
            NDarray x_train = np.array(x_train_list.ToArray());
            NDarray x_test = np.array(x_test_list.ToArray());
            NDarray y_train = np.array(y_train_list.ToArray());
            NDarray y_test = np.array(y_test_list.ToArray());

            var model = new Sequential();
            var opt = new  Keras.Optimizers.Adam(lr: (float)0.00001);
            model.Add(new Dense(10, activation: "relu", input_dim: 1)); //input_shape: new Shape(2)));
            model.Add(new Dense(10, activation: "relu"));
            model.Add(new Dense(1, activation: "relu"));

            model.Compile(optimizer: opt, loss: "mse", metrics: new string[] { "accuracy","mse" });
            model.Fit(x_train, y_train, batch_size: 64, epochs: 300);

            var predict = model.Predict(x_test);

            var line_predict = new OxyPlot.Series.LineSeries()
            {
                Title = $"Predict",
                Color = OxyPlot.OxyColors.Blue,
                StrokeThickness = 1,
                MarkerSize = 2,
                MarkerType = OxyPlot.MarkerType.Circle
            };
            var line_test = new OxyPlot.Series.LineSeries()
            {
                Title = $"Test",
                Color = OxyPlot.OxyColors.Red,
                StrokeThickness = 1,
                MarkerSize = 2,
                MarkerType = OxyPlot.MarkerType.Circle
            };
            var line_irr = new OxyPlot.Series.LineSeries()
            {
                Title = $"Test",
                Color = OxyPlot.OxyColors.Red,
                StrokeThickness = 1,
                MarkerSize = 2,
                MarkerType = OxyPlot.MarkerType.Circle
            };

            for (int i = 0;i<size_cont*0.2;i++)
            {
                line_predict.Points.Add(new OxyPlot.DataPoint(i, (double)predict[i]));
                line_test.Points.Add(new OxyPlot.DataPoint( i, (double)y_test[i]));
                line_irr.Points.Add(new OxyPlot.DataPoint(i, (double)x_test[i]));
            }

            // create the model and add the lines to it
            var plot = new OxyPlot.PlotModel
            {
                Title = $"Scatter Plot ({size_cont:N0} points each)"
            };
            plot.Series.Add(line_predict);
            plot.Series.Add(line_test);
            plot.Series.Add(line_irr);

            // load the model into the user control
            string filename = "D:/Documentos/Nuevas descargas/predict.png";
            var pngExporter = new PngExporter { Width = 800, Height = 400, Background = OxyColors.White };
            pngExporter.ExportToFile(plot, filename);



            Console.WriteLine(predict);
            Console.WriteLine("");

        }
    }

    public class Data
    {
        public float IRR_filter { get; set; }
        public float POT_filter { get; set; }
    }
}





