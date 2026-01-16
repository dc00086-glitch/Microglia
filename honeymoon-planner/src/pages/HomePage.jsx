import { useHoneymoon } from '../context/HoneymoonContext';
import { MapPin, Calendar, Heart, Plane, Hotel, Camera } from 'lucide-react';
import { format, differenceInDays, parseISO } from 'date-fns';

// Curated high-quality images for popular destinations
const curatedImages = {
  'Paris': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=400&q=80',
  'Rome': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=400&q=80',
  'Barcelona': 'https://images.unsplash.com/photo-1583422409516-2895a77efded?w=400&q=80',
  'Santorini': 'https://images.unsplash.com/photo-1613395877344-13d4a8e0d49e?w=400&q=80',
  'Venice': 'https://images.unsplash.com/photo-1514890547357-a9ee288728e0?w=400&q=80',
  'Florence': 'https://images.unsplash.com/photo-1543429258-c5ca3ea2e8a5?w=400&q=80',
  'Amsterdam': 'https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=400&q=80',
  'London': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=400&q=80',
  'Prague': 'https://images.unsplash.com/photo-1519677100203-a0e668c92439?w=400&q=80',
  'Vienna': 'https://images.unsplash.com/photo-1516550893923-42d28e5677af?w=400&q=80',
  'Lisbon': 'https://images.unsplash.com/photo-1585208798174-6cedd86e019a?w=400&q=80',
  'Athens': 'https://images.unsplash.com/photo-1555993539-1732b0258235?w=400&q=80',
  'Dublin': 'https://images.unsplash.com/photo-1549918864-48ac978761a4?w=400&q=80',
  'Munich': 'https://images.unsplash.com/photo-1595867818082-083862f3d630?w=400&q=80',
  'Nice': 'https://images.unsplash.com/photo-1491166617655-0723a0999cfc?w=400&q=80',
  'Amalfi': 'https://images.unsplash.com/photo-1633321702518-7feccafb94d5?w=400&q=80',
  'Cinque Terre': 'https://images.unsplash.com/photo-1516483638261-f4dbaf036963?w=400&q=80',
  'Swiss Alps': 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=400&q=80',
  'New York': 'https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=400&q=80',
  'Tokyo': 'https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400&q=80',
  'Sydney': 'https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?w=400&q=80',
  'Dubai': 'https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=400&q=80',
  'Bali': 'https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=400&q=80',
  'Maldives': 'https://images.unsplash.com/photo-1514282401047-d79a71a590e8?w=400&q=80',
  'Hawaii': 'https://images.unsplash.com/photo-1507876466758-bc54f384809c?w=400&q=80',
  'Cancun': 'https://images.unsplash.com/photo-1510097467424-192d713fd8b2?w=400&q=80',
  'Maui': 'https://images.unsplash.com/photo-1542259009477-d625272157b7?w=400&q=80',
  'Fiji': 'https://images.unsplash.com/photo-1584466990297-4ce1cd706d44?w=400&q=80',
  'Tahiti': 'https://images.unsplash.com/photo-1589197331516-4d84b72ebde3?w=400&q=80',
  'Bora Bora': 'https://images.unsplash.com/photo-1568359858554-6c99c0b4efe9?w=400&q=80',
  'Seychelles': 'https://images.unsplash.com/photo-1589979481223-deb893043163?w=400&q=80',
  'Mauritius': 'https://images.unsplash.com/photo-1586500036706-41963de24d8b?w=400&q=80',
  'Bangkok': 'https://images.unsplash.com/photo-1508009603885-50cf7c579365?w=400&q=80',
  'Singapore': 'https://images.unsplash.com/photo-1525625293386-3f8f99389edd?w=400&q=80',
  'Hong Kong': 'https://images.unsplash.com/photo-1536599018102-9f803c140fc1?w=400&q=80',
  'Madrid': 'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?w=400&q=80',
  'Berlin': 'https://images.unsplash.com/photo-1560969184-10fe8719e047?w=400&q=80',
  'Budapest': 'https://images.unsplash.com/photo-1551867633-194f125bddfa?w=400&q=80',
  'Copenhagen': 'https://images.unsplash.com/photo-1513622470522-26c3c8a854bc?w=400&q=80',
  'Stockholm': 'https://images.unsplash.com/photo-1509356843151-3e7d96241e11?w=400&q=80',
  'Edinburgh': 'https://images.unsplash.com/photo-1506377585622-bedcbb5f5ec3?w=400&q=80',
  'Bruges': 'https://images.unsplash.com/photo-1491557345352-5929e343eb89?w=400&q=80',
  'Salzburg': 'https://images.unsplash.com/photo-1594761450125-ca8aa1cc2b3f?w=400&q=80',
  'Dubrovnik': 'https://images.unsplash.com/photo-1555990538-1e6c4d9a0c3d?w=400&q=80',
  'Mykonos': 'https://images.unsplash.com/photo-1601581875039-e899893d520c?w=400&q=80',
  'Capri': 'https://images.unsplash.com/photo-1534113414509-0eec2bfb493f?w=400&q=80',
  'Lake Como': 'https://images.unsplash.com/photo-1540575861501-7cf05a4b125a?w=400&q=80',
  'Positano': 'https://images.unsplash.com/photo-1534113414509-0eec2bfb493f?w=400&q=80',
};

// Default fallback images for unknown cities (rotating beautiful travel photos)
const fallbackImages = [
  'https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400&q=80', // Travel map
  'https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=400&q=80', // Road trip
  'https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=400&q=80', // Lake mountains
  'https://images.unsplash.com/photo-1530789253388-582c481c54b0?w=400&q=80', // Travel adventure
  'https://images.unsplash.com/photo-1504150558240-0b4fd8946624?w=400&q=80', // Passport travel
];

// Generate image URL for any city - uses curated image if available, otherwise uses fallback
const getCityImage = (city) => {
  // Check for exact match first
  if (curatedImages[city]) {
    return curatedImages[city];
  }

  // Check for case-insensitive match
  const cityLower = city.toLowerCase();
  const matchedKey = Object.keys(curatedImages).find(key => key.toLowerCase() === cityLower);
  if (matchedKey) {
    return curatedImages[matchedKey];
  }

  // For unknown cities, use a consistent fallback based on city name
  // This ensures the same city always gets the same image
  const index = city.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % fallbackImages.length;
  return fallbackImages[index];
};

export default function HomePage({ setCurrentPage, goToItinerary }) {
  const { tripInfo, itinerary, bookings, scrapbook, getTotalBudget } = useHoneymoon();

  const daysUntilTrip = differenceInDays(parseISO(tripInfo.startDate), new Date());
  const tripDuration = differenceInDays(parseISO(tripInfo.endDate), parseISO(tripInfo.startDate)) + 1;
  const totalActivities = itinerary.reduce((sum, day) => sum + day.activities.length, 0);

  return (
    <div className="home-page">
      <header className="hero-section">
        <div className="hero-content">
          <h1>
            <Heart className="heart-icon" fill="#ff6b6b" />
            {tripInfo.couple}'s European Honeymoon
          </h1>
          <p className="hero-subtitle">
            {tripDuration} days of adventure, romance, and unforgettable memories
          </p>
          <div className="trip-dates">
            <Calendar size={18} />
            <span>
              {format(parseISO(tripInfo.startDate), 'MMMM d')} - {format(parseISO(tripInfo.endDate), 'MMMM d, yyyy')}
            </span>
          </div>
          {daysUntilTrip > 0 && (
            <div className="countdown">
              <span className="countdown-number">{daysUntilTrip}</span>
              <span className="countdown-label">days until your adventure begins!</span>
            </div>
          )}
        </div>
      </header>

      <section className="destinations-section">
        <h2>Your Destinations</h2>
        <p className="destinations-hint">Click a destination to see its itinerary</p>
        <div className="destination-cards-grid">
          {tripInfo.destinations.map((city, index) => {
            const daysInCity = itinerary.filter(day => day.city === city).length;
            return (
              <div
                key={city}
                className="destination-card-img"
                style={{ animationDelay: `${index * 0.1}s` }}
                onClick={() => goToItinerary(city)}
              >
                <img src={getCityImage(city)} alt={city} />
                <div className="destination-overlay">
                  <MapPin size={18} />
                  <span>{city}</span>
                  {daysInCity > 0 && <span className="days-badge">{daysInCity} days</span>}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="stats-section">
        <div className="stat-card" onClick={() => setCurrentPage('itinerary')}>
          <Calendar size={32} />
          <div className="stat-info">
            <span className="stat-number">{itinerary.length}</span>
            <span className="stat-label">Days Planned</span>
          </div>
        </div>
        <div className="stat-card" onClick={() => setCurrentPage('itinerary')}>
          <MapPin size={32} />
          <div className="stat-info">
            <span className="stat-number">{totalActivities}</span>
            <span className="stat-label">Activities</span>
          </div>
        </div>
        <div className="stat-card" onClick={() => setCurrentPage('bookings')}>
          <Plane size={32} />
          <div className="stat-info">
            <span className="stat-number">{bookings.length}</span>
            <span className="stat-label">Bookings</span>
          </div>
        </div>
        <div className="stat-card" onClick={() => setCurrentPage('scrapbook')}>
          <Camera size={32} />
          <div className="stat-info">
            <span className="stat-number">{scrapbook.length}</span>
            <span className="stat-label">Memories</span>
          </div>
        </div>
      </section>

      <section className="budget-section">
        <h2>Trip Budget</h2>
        <div className="budget-card">
          <div className="budget-amount">
            <span className="currency">$</span>
            <span className="amount">{getTotalBudget().toLocaleString()}</span>
          </div>
          <p className="budget-label">Total Booked</p>
          <div className="budget-breakdown">
            <div className="budget-item">
              <Plane size={16} />
              <span>Flights: ${bookings.filter(b => b.type === 'flight').reduce((s, b) => s + (b.cost || 0), 0).toLocaleString()}</span>
            </div>
            <div className="budget-item">
              <Hotel size={16} />
              <span>Hotels: ${bookings.filter(b => b.type === 'hotel').reduce((s, b) => s + (b.cost || 0), 0).toLocaleString()}</span>
            </div>
            <div className="budget-item">
              <MapPin size={16} />
              <span>Activities: ${bookings.filter(b => b.type === 'activity').reduce((s, b) => s + (b.cost || 0), 0).toLocaleString()}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="quick-actions">
        <h2>Quick Actions</h2>
        <div className="action-buttons">
          <button className="action-btn" onClick={() => setCurrentPage('itinerary')}>
            <Calendar size={20} />
            Plan Your Days
          </button>
          <button className="action-btn" onClick={() => setCurrentPage('bookings')}>
            <Plane size={20} />
            Manage Bookings
          </button>
          <button className="action-btn" onClick={() => setCurrentPage('scrapbook')}>
            <Camera size={20} />
            Add Memories
          </button>
        </div>
      </section>
    </div>
  );
}
