#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

Sun::Sun(AudioContainer& audiocontainer_ref, Parameters& parameters_ref, ControlPanel& controlpanel_ref)
    :   Astro(audiocontainer_ref, parameters_ref, controlpanel_ref){
    Logger::writeToLog("Sun created.");
}

Sun::~Sun(){
    Logger::writeToLog("Planet destroyed.");
}

//------------------------------------------------------------//
// View methods.

void Sun::paint(Graphics& g){
    g.setColour(juce::Colours::white);
    g.fillEllipse(0, 0, getDiameter(), getDiameter());
}

void Sun::resized(){
    draw();
    setPosXY(getX(), getY());
}

void Sun::draw(){
    setSize(getDiameter(), getDiameter());
    setCentreRelative(0.5, 0.5);
}

void Sun::draw(int diameter, int x, int y){setBounds(x, y, diameter, diameter);}


//------------------------------------------------------------//
// Interface methods.

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
