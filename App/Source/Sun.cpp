#include "Headers.h"


// Sun class which inherits from the parent Planet class.
// Movement is disabled and position is fixed.


//------------------------------------------------------------//
// Constructors and destructors.

Sun::Sun(){}

Sun::Sun(juce::OwnedArray<Planet>* planets_ptr, AudioContainer* audiocontainer_ptr, juce::ValueTree state)
    : Planet(planets_ptr, audiocontainer_ptr, state){}

Sun::~Sun(){}


//------------------------------------------------------------//
// View methods.

void Sun::paint(Graphics& g){
    g.setColour(juce::Colours::yellow);
    g.fillEllipse(0, 0, getDiameter(), getDiameter());
}

void Sun::resized(){}

void Sun::draw(){
    setSize(getDiameter(), getDiameter());
    setCentreRelative(0.5, 0.5);
}

//------------------------------------------------------------//
// Interface methods.

int Sun::getDiameter(){return m_State.getProperty(Parameters::diameterProp);}


//------------------------------------------------------------//
// Temporary methods.

void Sun::generateLatents(){m_Latents = Generator::generateLatents();}
void Sun::generateSample(at::Tensor& latents){m_Sample = Generator::generateSample(latents);}

//------------------------------------------------------------//
// Controller methods.

bool Sun::hitTest(int x, int y){
    float a = pow((float)x - ((float)getDiameter() / 2.0f), 2.0f);
    float b = pow((float)y - ((float)getDiameter() / 2.0f), 2.0f);
    float c = sqrt(a + b);

    return c <= (float)(getDiameter() / 2);
}

void Sun::mouseDown(const MouseEvent& e){
    if(e.getNumberOfClicks() > 1 && e.mods.isLeftButtonDown()){
        Logger::writeToLog("Generating sample...");

        generateLatents();
        generateSample(m_Latents);

        Logger::writeToLog("Sample generated.");
    }

    else if(e.mods.isLeftButtonDown() && e.mouseWasClicked()){
        addSample();
        playSample();
    }
}

void Sun::mouseUp(const MouseEvent& e){}
void Sun::mouseDrag(const MouseEvent& e){}
void Sun::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){}
