#include "Headers.h"


#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Astro::Astro(AudioContainer& audiocontainer_ref, Parameters& parameters_ref, ControlPanel& controlpanel_ref)
    :   m_AudioContainerRef(audiocontainer_ref), m_ParametersRef(parameters_ref), m_ControlPanelRef(controlpanel_ref){}

Astro::~Astro(){}

//--------------------------------------------------//
// Interface methods.

void Astro::setDiameter(int diameter){getState().setProperty(Parameters::diameterProp, diameter, nullptr);}

void Astro::setPosXY(int x, int y){
    getState().setProperty(Parameters::posXProp, x, nullptr);
    getState().setProperty(Parameters::posYProp, y, nullptr);
    setCentrePosXY(x + getDiameter() / 2, y + getDiameter() / 2);
}

void Astro::setCentrePosXY(int x, int y){
    getState().setProperty(Parameters::posCentreXProp, x, nullptr);
    getState().setProperty(Parameters::posCentreYProp, y, nullptr);
}

int Astro::getDiameter(){return getState().getProperty(Parameters::diameterProp);}
int Astro::getPosX(){return getState().getProperty(Parameters::posXProp);}
int Astro::getPosY(){return getState().getProperty(Parameters::posYProp);}
int Astro::getCentreX(){return getState().getProperty(Parameters::posCentreXProp);}
int Astro::getCentreY(){return getState().getProperty(Parameters::posCentreYProp);}

float Astro::getDistance(int xa, int ya, int xb, int yb){
    float a = (float)pow(xb - xa, 2);
    float b = (float)pow(yb - ya, 2); 
    return sqrt(a + b);
}

float Astro::getDistance(Astro* astro_a, Astro* astro_b){
    int centreXA = astro_a->getCentreX();
    int centreYA = astro_a->getCentreY();
    int centreXB = astro_b->getCentreX();
    int centreYB = astro_b->getCentreY();

    float a = (float)pow(centreXB - centreXA, 2);
    float b = (float)pow(centreYB - centreYA, 2);

    return sqrt(a + b);
}

void Astro::updateGraph(){getState().setProperty(Parameters::updateGraphSignal, true, nullptr);}
void Astro::generateSample(){getState().setProperty(Parameters::generateSampleSignal, true, nullptr);}

void Astro::playSample(){
    Logger::writeToLog("Playing audio...");
    m_ParametersRef.setActivePlanet(getState());
    m_AudioContainerRef.sampleIndex.clear();
    m_AudioContainerRef.playAudio = true;
}

//--------------------------------------------------//
// Controller methods.

void Astro::mouseEnter(const MouseEvent& e){
    juce::ignoreUnused(e);
    m_ControlPanelRef.show(getState());
}

void Astro::mouseExit(const MouseEvent& e){
    juce::ignoreUnused(e);
    m_ControlPanelRef.unshow();
}

void Astro::valueChanged(juce::Value& value){
    juce::ignoreUnused(value);
    repaint();
}

//--------------------------------------------------//
// Animator class.

//--------------------------------------------------//
// Constructors and destructors.

Astro::Animator::Animator(){
    m_DiameterShift.setValue(0);
    startTimer(17);
}

Astro::Animator::~Animator(){
    stopTimer();
}

//--------------------------------------------------//
// Interface methods.

int Astro::Animator::getDiameterShift(){return (int)m_DiameterShift.getValue();}

//--------------------------------------------------//
// Callback methods.

void Astro::Animator::timerCallback(){
    if((int)m_DiameterShift.getValue() >= 20){
        m_DiameterShiftDirection = false;
    }
    else if((int)m_DiameterShift.getValue() <= -20){
        m_DiameterShiftDirection = true;
    }

    if(m_DiameterShiftDirection == true){
        m_DiameterShift.setValue((int)m_DiameterShift.getValue() + 1);
    }
    else{
        m_DiameterShift.setValue((int)m_DiameterShift.getValue() - 1);
    }
}
