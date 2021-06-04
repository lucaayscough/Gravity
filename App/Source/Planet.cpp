#include "Headers.h"


// Main Planet class.

//--------------------------------------------------//
// Constructors and destructors.

Planet::Planet(){}

Planet::Planet(juce::OwnedArray<Planet>* planets_ptr, AudioContainer* audiocontainer_ptr, Parameters* parameters_ptr):
    m_PlanetsPtr(planets_ptr),
    m_AudioContainerPtr(audiocontainer_ptr),
    m_ParametersPtr(parameters_ptr){
    allocateStorage();

    // Generate random sample.
    generateLatents();
    generateSample(m_Latents);

    // Listener value used to determine when to destroy the planet.
    m_Destroy.setValue(false);

    // Listener used to detect when lerp graph needs recalculating.
    m_LerpGraph.setValue(false);

    setPosXY(getX(), getY());
}

Planet::~Planet(){}


//--------------------------------------------------//
// Public methods.

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    draw(m_Diameter, getX(), getY());
    g.fillEllipse(m_ClipBoundary / 2, m_ClipBoundary / 2, m_Diameter, m_Diameter);
}

void Planet::resized(){}

void Planet::draw(int diameter, int x, int y){
    setBounds(x, y, diameter + m_ClipBoundary, diameter + m_ClipBoundary);
}

void Planet::resizePlanet(int diameter){
    int new_x;
    int new_y;

    if(diameter > getDiameter()){
        new_x = getX() - (M_SIZE_MODIFIER / 2);
        new_y = getY() - (M_SIZE_MODIFIER / 2);
    } else{
        new_x = getX() + (M_SIZE_MODIFIER / 2);
        new_y = getY() + (M_SIZE_MODIFIER / 2);
    }

    setDiameter(diameter);
    draw(diameter, new_x, new_y);

    // NEED A WAY TO UPDATE GRAPH WHEN DONE ZOOMING IN ON SOUND
    //updateGraph();
}

void Planet::setDiameter(int diameter){
    m_Diameter = diameter;
}

void Planet::setMapBoundaries(int width, int height){
    m_MapWidth = width;
    m_MapHeight = height;
}

void Planet::setPosXY(int x, int y){
    m_PosX = x;
    m_PosY = y;
}

int Planet::getDiameter(){
    return m_Diameter;
}

int Planet::getClipBoundary(){
    return m_ClipBoundary;
}

float Planet::getDistance(int xa, int ya, int xb, int yb){  
    float a = (float)pow(xb - xa, 2);
    float b = (float)pow(yb - ya, 2); 
    return sqrt(a + b);
}

float Planet::getDistance(Planet* planet_a, Planet* planet_b){  
    int centreXA = getCentreX(planet_a);
    int centreYA = getCentreY(planet_a);
    int centreXB = getCentreX(planet_b);
    int centreYB = getCentreY(planet_b);

    float a = (float)pow(centreXB - centreXA, 2);
    float b = (float)pow(centreYB - centreYA, 2);

    return sqrt(a + b);
}

int Planet::getCentreX(Planet* planet){
    return planet->getX() + ((planet->getDiameter() + planet->getClipBoundary()) / 2);
}

int Planet::getCentreY(Planet* planet){
    return planet->getY() + ((planet->getDiameter() + planet->getClipBoundary()) / 2);
}

void Planet::generateLatents(){m_Latents = Generator::generateLatents();}
void Planet::generateSample(at::Tensor& latents){m_Sample = Generator::generateSample(latents);}

void Planet::updateGraph(){
    m_LerpGraph.setValue(true);
    m_LerpGraph.setValue(false);
}

void Planet::addSample(){
    m_AudioContainerPtr->audio.clear();
    m_AudioContainerPtr->audio.addArray(m_Sample);
}

void Planet::playSample(){
    Logger::writeToLog("Playing audio...");
    m_AudioContainerPtr->sampleIndex.clear();
    m_AudioContainerPtr->playAudio = true;
}

//--------------------------------------------------//
// Private methods.

bool Planet::hitTest(int x, int y){
    float a = pow((float)x - ((float)m_Diameter + (float)m_ClipBoundary) / 2.0f, 2.0f);
    float b = pow((float)y - ((float)m_Diameter + (float)m_ClipBoundary) / 2.0f, 2.0f);
    return sqrt(a + b) <= m_Diameter / 2;
}

void Planet::mouseDown(const MouseEvent& e){
    // Starts dragging component.
    m_Dragger.startDraggingComponent(this, e);
}

void Planet::mouseUp(const MouseEvent& e){
    if(e.mods.isLeftButtonDown()){
        // Generates new sample if double clicked with left mouse button.
        if(e.getNumberOfClicks() > 1){
            Logger::writeToLog("Generating sample...");

            generateLatents();
            generateSample(m_Latents);

            Logger::writeToLog("Sample generated.");
        }
        
        // Plays sample if clicked once with left mouse button.
        else if(e.getNumberOfClicks() == 1 && e.mouseWasClicked()){
            addSample();
            playSample();
        }

        else if(e.mouseWasDraggedSinceMouseDown()){
            updateGraph();
        }
    }
    
    // Destroys planet if clicked with right mouse button.
    else if(e.mods.isRightButtonDown()){
        // Initializes planet destruction.
        m_Destroy.setValue(true);
        Logger::writeToLog("Set to destroy.");
    }
}

void Planet::mouseDrag(const MouseEvent& e){
    m_Dragger.dragComponent(this, e, nullptr);
    checkCollision();
    checkBounds();
    setPosXY(getX(), getY());
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){    
    Logger::writeToLog("Wheel moved.");

    if(w.deltaY > 0.0f && m_Diameter < M_MAX_PLANET_SIZE){
        resizePlanet(m_Diameter + M_SIZE_MODIFIER);
    }
    else if(w.deltaY < 0.0f && m_Diameter > M_MIN_PLANET_SIZE){
        resizePlanet(m_Diameter - M_SIZE_MODIFIER);
    }
}

void Planet::visibilityChanged(){
    Logger::writeToLog("Visibility changed.");
}

void Planet::checkCollision(){
    int centrePosX = getCentreX(this);
    int centrePosY = getCentreY(this);

    float distance, minDistance;

    // Check collision with sun.
    {
        int centreXSun = Variables::WINDOW_WIDTH / 2;
        int centreYSun = Variables::WINDOW_HEIGHT / 2;
        int sunDiameter = Variables::SUN_DIAMETER;

        distance = getDistance(centrePosX, centrePosY, centreXSun, centreYSun);
        minDistance = (sunDiameter + m_Diameter) / 2;

        if(distance <= minDistance){
            draw(m_Diameter, m_PosX, m_PosY);
        }
    }

    Planet* planet;
    int centrePosX2, centrePosY2;

    for(int i = 0; i < m_PlanetsPtr->size(); i++){
        // Variable for ease of use.
        planet = (*m_PlanetsPtr)[i];

        // Avoid self collision testing.
        if(planet->getComponentID() != getComponentID()){
            centrePosX2 = getCentreX(planet);
            centrePosY2 = getCentreY(planet);

            distance = getDistance(centrePosX, centrePosY, centrePosX2, centrePosY2);
            minDistance = (planet->getDiameter() + m_Diameter) / 2;

            if(distance <= minDistance){
                draw(m_Diameter, m_PosX, m_PosY);
            }
        }
    }
}

void Planet::checkBounds(){
    //Check left boundary.
    if(getX() < -(m_ClipBoundary / 2))
        draw(m_Diameter, -(m_ClipBoundary / 2), getY());

    // Check top boundary.
    if(getY() < -(m_ClipBoundary / 2))
        draw(m_Diameter, getX(), -(m_ClipBoundary / 2));

    // Check right boundary,
    if(getX() + m_Diameter + (m_ClipBoundary / 2) > m_MapWidth)
        draw(m_Diameter, m_MapWidth - m_Diameter - (m_ClipBoundary / 2), getY());

    // Check bottom boundary.
    if(getY() + m_Diameter + (m_ClipBoundary / 2) > m_MapHeight)
        draw(m_Diameter, getX(), m_MapHeight - m_Diameter - (m_ClipBoundary / 2));
    
    // Write planet position to screen.
    //Logger::writeToLog("X: " + std::to_string(getX()) + ", Y: " + std::to_string(getY()));
}

void Planet::allocateStorage(){m_Sample.ensureStorageAllocated(Generator::M_NUM_SAMPLES);}
