#include "Headers.h"


// Sun class which inherits from the parent Planet class.
// Movement is disabled and position is fixed.

Sun::Sun(){}

Sun::Sun(juce::OwnedArray<Planet>* planets_ptr, Generator* generator_ptr, AudioContainer* audiocontainer_ptr)
    : Planet(planets_ptr, generator_ptr, audiocontainer_ptr){}

Sun::~Sun(){}

void Sun::paint(Graphics& g){
    g.setColour(juce::Colours::yellow);
    g.fillEllipse(0, 0, M_DIAMETER, M_DIAMETER);
}

void Sun::resized(){}

void Sun::draw(){
    setSize(M_DIAMETER, M_DIAMETER);
    setCentreRelative(0.5, 0.5);
}

int Sun::getDiameter(){
    return M_DIAMETER;
}

void Sun::generateLatents(){
    m_Latents = m_GeneratorPtr->generateLatents();
}

void Sun::generateSample(at::Tensor& latents){
    m_Sample = m_GeneratorPtr->generateSample(latents);
}

bool Sun::hitTest(int x, int y){
    float a = pow((float)x - ((float)M_DIAMETER / 2.0f), 2.0f);
    float b = pow((float)y - ((float)M_DIAMETER / 2.0f), 2.0f);
    float c = sqrt(a + b);

    return c <= (float)(M_DIAMETER / 2);
}

void Sun::mouseDown(const MouseEvent& e){
    if(e.getNumberOfClicks() > 1 && e.mods.isLeftButtonDown()){
        Logger::writeToLog("Generating sample...");

        generateLatents();
        generateSample(m_Latents);

        Logger::writeToLog("Sample generated.");
    }

    else if(e.mods.isLeftButtonDown() && e.mouseWasClicked()){
        playSample();
    }
}

void Sun::mouseUp(const MouseEvent& e){}

void Sun::mouseDrag(const MouseEvent& e){}

void Sun::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){}
