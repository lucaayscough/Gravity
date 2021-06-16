#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

Sun::Sun(juce::OwnedArray<Planet>& planets_ref, AudioContainer& audiocontainer_ref, Parameters& parameters_ref, ControlPanel& controlpanel_ref)
    :   Planet(planets_ref, audiocontainer_ref, parameters_ref, controlpanel_ref){}

void Sun::init(){
    Logger::writeToLog("Sun created.");
    setComponentEffect(&m_GlowEffect);
}

Sun::~Sun(){
    setComponentEffect(nullptr);
    Logger::writeToLog("Sun destroyed.");
}

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

void Sun::setPosXY(int x, int y){
    getState().setProperty(Parameters::posXProp, x, nullptr);
    getState().setProperty(Parameters::posYProp, y, nullptr);
    setCentrePosXY(x + getDiameter() / 2, y + getDiameter() / 2);
}

juce::ValueTree Sun::getState(){return m_ParametersRef.getSunNode();}

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
        generateSample();
    }

    else if(e.mods.isLeftButtonDown() && e.mouseWasClicked()){
        playSample();
    }
}

void Sun::mouseUp(const MouseEvent& e){juce::ignoreUnused(e);}
void Sun::mouseDrag(const MouseEvent& e){juce::ignoreUnused(e);}
void Sun::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){juce::ignoreUnused(e, w);}
