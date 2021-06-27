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

    g.fillEllipse(
        getClipBoundary() / 2,
        getClipBoundary() / 2,
        getDiameter(),
        getDiameter() 
    );
}

void Sun::resized(){
    draw();
    setPosXY(getX(), getY());
}

void Sun::draw(){
    setSize(getDiameterWithClipBoundary(), getDiameterWithClipBoundary());
    setCentreRelative(0.5, 0.5);
}

//------------------------------------------------------------//
// Interface methods.

juce::ValueTree Sun::getState(){return m_ParametersRef.getSunNode();}

//------------------------------------------------------------//
// Controller methods.

void Sun::mouseDown(const MouseEvent& e){
    if(e.getNumberOfClicks() > 1 && e.mods.isLeftButtonDown()){
        generateSample();
    }

    else if(e.mods.isLeftButtonDown() && e.mouseWasClicked()){
        playSample();
    }
}
