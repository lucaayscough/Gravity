#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

Sun::Sun(juce::OwnedArray<Planet>& planets_ref, AudioContainer& audiocontainer_ref, Parameters& parameters_ref)
    : Planet(planets_ref, audiocontainer_ref, parameters_ref){}

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

void Sun::setPosXY(int, int){}

juce::ValueTree Sun::getState(){return m_ParametersRef.getSunNode();}
int Sun::getDiameter(){return getState().getProperty(Parameters::diameterProp);}
int Sun::getCentreX(Planet* planet){return planet->getX() + (planet->getDiameter() / 2);}
int Sun::getCentreY(Planet* planet){return planet->getY() + (planet->getDiameter() / 2);}

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
