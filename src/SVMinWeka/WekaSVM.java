/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SVMinWeka;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Enumeration;
import java.net.URL;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
import weka.classifiers.functions.SMO;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.SerializationHelper;
import weka.clusterers.EM; /* Expectation-Maximization */

//import wlsvm.WLSVM;

public class WekaSVM {
    
    static BufferedReader bfrdr;
    SMO modelSMO;
    
    public WekaSVM() throws Exception{
        bfrdr = new BufferedReader(new InputStreamReader(System.in));
        modelSMO = null;
        
    }
    public BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;
        try {
            inputReader = new BufferedReader(new FileReader(filename));
            
        } 
        catch (FileNotFoundException ex) 
        {
            System.err.println("File Not Found:"+filename);
        }
        
        return inputReader;
    }
    
    public Evaluation simpleClassify(Classifier model, Instances trainingSet, Instances testingSet) 
            throws Exception {
        
            Evaluation validation = new Evaluation(trainingSet);
            model.buildClassifier(trainingSet);
            validation.evaluateModel(model, testingSet);
            
            return validation;
    }
    
    
    public double calculateAccuracy(FastVector predictions) {
        double correct = 0;
        
        for (int i =0; i<predictions.size();i++) {
            NominalPrediction np = (NominalPrediction)predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }
        
        return 100 * correct/predictions.size();
    }
    
    public Instances[][] crossValidationSplit(Instances data, int numberofFolds) {
        Instances[][] split = new Instances[2][numberofFolds];
        
        for (int i=0; i<numberofFolds;i++) {
            split[0][i] = data.trainCV(numberofFolds, i);
            split[1][i] = data.testCV(numberofFolds, i);
            
        }
        
        return split;
    }
    
    /*public void SVMClassify(String datafile) throws Exception {
        WLSVM svm = new WLSVM();
        
        String[] ops = { "-t", datafile, "-x", "5", "-i", "-S", "0", "-K", "2",  
             "-G", "-G", "1", "-C", "7", "-Z", "1", "-M", "100" };
        
        try {
        System.out.println(svm.getOptions());
        System.out.println(Evaluation.evaluateModel(svm, ops));
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }
    }*/
    
    public void getSavedModel() throws Exception{
        	
	//InputStream classModelStream;
	String modelfile, ch;
		
        System.out.print("Do you want to get a saved model (Yes/No) ?");        
        ch = bfrdr.readLine();
        System.out.println();
        //String currentDir = System.getProperty("user.dir");
        //System.out.println("Current dir using System:" +currentDir);
        if (ch.equalsIgnoreCase("Yes")) {
            System.out.print("Enter saved model file name :");            
            modelfile = bfrdr.readLine();
            System.out.println();
            //classModelStream =getClass().getClassLoader().getResourceAsStream(modelfile);
            /*URL url = getClass().getResource(currentDir +"\\" + modelfile);
            classModelStream = url.openStream(); 
            if (classModelStream == null)
                System.out.println("Error:Can not open the file");
            else*/
            //{
                //System.out.println("Total Bytes = "+classModelStream.available());
                modelSMO = (SMO)SerializationHelper.read(modelfile);//classModelStream);
                if (modelSMO == null)
                    System.out.println("Error:Can not load the model file!");
                else
                    System.out.println("Successfully loaded the model file!");
            //}
        }		
    }
    
    public void classifyAnInstance(String datafilename) throws Exception {
        if (modelSMO == null)
        {
            System.out.println("Error:modelSMO object is not loaded!");
        } 
        else {
             BufferedReader datafile = readDataFile(datafilename);
             Instances data = new Instances(datafile);
             boolean notExit = true;
             int index;
             data.setClassIndex(data.numAttributes()-1);
             
             while (notExit) {
                 System.out.print("Enter index to classify in (0- to exit)"+datafilename+ " : ");            
                 index = Integer.parseInt(bfrdr.readLine());
                 if (index <= 0)
                 {
                     notExit = false;
                 }
                 else
                 { 
                    //System.out.println("Classes : " + data.classAttribute().value(0)+" and "+data.classAttribute().value(1));
                    System.out.println("Data at index = "+index+" is   "+data.instance(index-1).toString());
                    System.out.println("The class for index "+index+" is :"+data.classAttribute().value((int)modelSMO.classifyInstance(data.instance(index-1))));
                 }
             }
                 
        }
        
        
    }
    
    public void SMOClassify(String datafilename, int saveModel) throws Exception {
        BufferedReader datafile = readDataFile(datafilename);//"iris.arff");
        
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes()-1);
        
        //Choose a type of validation split
        Instances[][] split = crossValidationSplit(data, 10);        
        //Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];
        
        SMO polysmo = new SMO();
        SMO rbfsmo = new SMO();
        RBFKernel rbfkernel = new RBFKernel();
        PolyKernel polykernel = new PolyKernel();
        double Cparam, Gparam;
        double bestAccuracy = 0, bestC = 0, bestG = 0;
        FastVector predictions = new FastVector();
        
        //polysmo.classifyInstance(null);
        /*System.out.println();
        System.out.print("RBF Kernel Options : ");
           for (Enumeration e = kernel1.listOptions(); e.hasMoreElements();)
                        System.out.print(e.nextElement());
        System.out.println();
        System.out.print("Poly Kernel Options : ");
           for (Enumeration e = kernel.listOptions(); e.hasMoreElements();)
                        System.out.print(e.nextElement());
        System.out.println();
         */
        
        polykernel.buildKernel(data);
        polysmo.setKernel(polykernel);
        //rbfkernel.setGamma(Math.pow(2, Gparam));        
        rbfkernel.buildKernel(data);
        
        rbfsmo.setKernel(rbfkernel);
        //Choose a set of classifier
        //Classifier[] models = {polysmo};//new SMO()};
        //Classifier model = rbfsmo;//polysmo;//new SMO()};
        //, new WLSVM()}; //new LibSVM()};
        //LibSVM svm = new LibSVM();
        //WLSVM wsvm = new WLSVM();       
       //System.out.println("SMO Options : "+smo.);
                
        //polysmo.setNumFolds(10);
        rbfsmo.setNumFolds(10);
        Cparam =15;//3
        Gparam = 5; //-7
        //Run for each classifier model
        for (int j = 0; j<20;j++) {
            //Collect every group of predictions for current model in a FastVector            
            predictions.clear();
            //polysmo.setC(Cparam);           
            rbfsmo.setC(Math.pow(2, Cparam));
            rbfkernel.setGamma(Math.pow(2, Gparam));
            //For each training-testing split pair, traing and test the classifier
            for (int i = 0; i<trainingSplits.length;i++) {
                Evaluation validation = simpleClassify(rbfsmo, trainingSplits[i], testingSplits[i]);
                predictions.appendElements(validation.predictions());
                //See summary of each training-testing pair
                //System.out.println((j+1)+". "+models[j].toString());
                
            }
            
            //Calculate overall accuracy of current classifier on all splits;
            double accuracy = calculateAccuracy(predictions);
            
            if (bestAccuracy < accuracy)
            {
                bestAccuracy = accuracy;
                bestC = rbfsmo.getC();
                bestG = ((RBFKernel)rbfsmo.getKernel()).getGamma();
            }
           
            //Print current classifier's name & accuracy in a compicated and nice looking way
            System.out.println(rbfsmo.getClass().getSimpleName()+" : C = 2^"+Cparam+" Gamma = 2^"+Gparam+" Accuracy = "+String.format("%.2f%%", accuracy)+"\n====================");
            Cparam = Cparam + 1;
            Gparam = Gparam + 1;
        }
        System.out.println("Best C and Gamma for SMO/RBF Model are: "+bestC+" and "+bestG + " with accuracy "+String.format("%.2f%%", bestAccuracy));
        
        if (saveModel == 1)
        {
            predictions.clear();
            rbfsmo.setC(bestC);
            ((RBFKernel)rbfsmo.getKernel()).setGamma(bestG);            
            for (int i = 0; i<trainingSplits.length;i++) {
                Evaluation validation = simpleClassify(rbfsmo, trainingSplits[i], testingSplits[i]);
                predictions.appendElements(validation.predictions());
                //See summary of each training-testing pair
                //System.out.println((j+1)+". "+models[j].toString());
                
            }
            
            //Calculate overall accuracy of current classifier on all splits;
            double accuracy = calculateAccuracy(predictions);
            System.out.println("Proving the accuracy as "+String.format("%.2f%%", accuracy));
            System.out.print("Enter file name to save the mode:");            
            SerializationHelper.write(bfrdr.readLine(), rbfsmo);
            System.out.println();
        }
        
    }
    
    public void startMain() throws Exception
    {
        String ch;
        int saveModel = 0;        
        
        System.out.print("Enter Choice: Normal Classification from DataSet(1) or Classify an Instance based on the saved model(2) : ");
        ch = bfrdr.readLine();
        System.out.println();
        System.out.println("You made choice "+ch+"  .. Processing !");
        if (ch.equals("1"))
        {
            System.out.print("Save the best model (Yes/No) ?");
            ch = bfrdr.readLine();
            if (ch.equalsIgnoreCase("Yes"))
            {
                saveModel = 1;
            }
            SMOClassify("NugacheBot1Normal-MNJ1.1.arff", saveModel);
        }
        else if (ch.equals("2"))
        {                      
            getSavedModel();
            classifyAnInstance("NugacheBot1Normal-MNJ1.2.arff");
        }
    }
    public static void main(String[] args) throws Exception {
        
        WekaSVM svmobj = new WekaSVM();
        
        svmobj.startMain();
        //SVMClassify("iris.arff");        
    }
        
}

