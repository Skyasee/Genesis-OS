import asyncio
import logging
from genius_ai.core.genesis import GenesisOS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Initialize Genesis OS
        logger.info("Initializing Genesis OS...")
        genesis = GenesisOS()
        
        # Start the system
        logger.info("Starting Genesis OS...")
        await genesis.start()
        
        # Keep the system running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")
        finally:
            # Stop the system gracefully
            logger.info("Stopping Genesis OS...")
            await genesis.stop()
            
    except Exception as e:
        logger.error(f"Error running Genesis OS: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 