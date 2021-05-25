#include "Headers.h"


// Sun class which inherits from the parent Planet class.
// Movement is disabled and position is fixed.

Sun::Sun(){}

Sun::Sun(Generator* generator_ptr)
    : Planet(generator_ptr){}

Sun::~Sun(){}

void Sun::paint(Graphics& g){
    setSize(M_DIAMETER, M_DIAMETER);
    g.setColour(juce::Colours::yellow);
    g.fillEllipse(0, 0, M_DIAMETER, M_DIAMETER);
}

void Sun::resized(){}

int Sun::getDiameter(){
    return M_DIAMETER;
}

void Sun::generateLatents(){
    m_Latents = m_GeneratorPtr->generateLatents();
}

void Sun::generateSample(){
    m_Sample = m_GeneratorPtr->generateSample(m_Latents);
}

void Sun::mouseDown(const MouseEvent& e){
    if(e.getNumberOfClicks() > 1 && e.mods.isLeftButtonDown()){
        Logger::writeToLog("Generating sample...");

        generateLatents();
        generateSample();

        Logger::writeToLog("Sample generated.");
    }
}

void Sun::mouseDrag(const MouseEvent& e){}