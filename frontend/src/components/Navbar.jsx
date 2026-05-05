import { NavLink } from 'react-router-dom';
import { Droplets, TrendingUp, Database, Calendar, BarChart3, Menu, X } from 'lucide-react';
import { useState } from 'react';

const navLinks = [
  { to: '/', icon: <Database size={18} />, label: 'Dashboard' },
  { to: '/predict', icon: <Droplets size={18} />, label: 'Rainfall Predictor' },
  { to: '/storage', icon: <TrendingUp size={18} />, label: 'Water Storage' },
  { to: '/irrigation', icon: <Calendar size={18} />, label: 'Irrigation Scheduler' },
  { to: '/compare', icon: <BarChart3 size={18} />, label: 'Model Comparison' },
];

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="bg-dark-card/80 backdrop-blur-md border-b border-dark-border sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center gap-2">
            <Droplets className="text-brand-blue h-6 w-6" />
            <span className="font-bold text-xl tracking-tight">DK-AquaPredict</span>
          </div>
          
          <div className="hidden md:flex space-x-1">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive 
                      ? 'bg-brand-blue/20 text-brand-blue' 
                      : 'text-dark-muted hover:text-dark-text hover:bg-slate-700/30'
                  }`
                }
              >
                {link.icon}
                {link.label}
              </NavLink>
            ))}
          </div>

          <button className="md:hidden p-2" onClick={() => setIsOpen(!isOpen)}>
            {isOpen ? <X /> : <Menu />}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="md:hidden bg-dark-card border-t border-dark-border px-4 py-2 space-y-1">
          {navLinks.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              onClick={() => setIsOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
                  isActive ? 'bg-brand-blue/20 text-brand-blue' : 'text-dark-muted'
                }`
              }
            >
              {link.icon}
              {link.label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  );
}